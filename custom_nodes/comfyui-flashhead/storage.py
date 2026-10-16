"""Preflight checks for the persistent workspace: a Network Volume or a Global Volume."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

LOG = "[FlashHead Storage]"
WORKSPACE = Path("/workspace")
# Where RunPod mounts a Global Volume when a Network Volume takes /workspace, or when
# the deployer moves it there.
GLOBAL_ALTERNATE = Path("/workspace-global")
EPHEMERAL = ("overlay", "tmpfs", "ramfs")
# ComfyUI folders kept on a Global Volume, relative to the ComfyUI directory. Logs,
# temp files and ComfyUI-Manager's caches under user/ stay on the container disk.
GLOBAL_DATA = ("models", "input", "output", "user/default")


def object_filesystem(kind):
    """FUSE is an interface, not a storage type; POSIX network mounts use it too."""
    return any(name in kind.lower() for name in ("s3fs", "geesefs", "gcsfuse", "rclone", "fuse.global"))


def mount_kinds(path):
    """Filesystem types mounted exactly at path, from /proc/self/mountinfo."""
    path = Path(path).resolve()
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.exists():
        return []
    kinds = []
    for line in mountinfo.read_text().splitlines():
        fields = line.split()
        target = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
        if target == path:
            kinds.append(fields[fields.index("-") + 1].lower())
    return kinds


def hard_links_work(path):
    """Global Volumes are object-backed: no hard links, atomic rename, locks or modes.

    Mount metadata does not name them reliably, so test the one operation RunPod
    documents as missing. A volume that cannot be written fails here with the reason.
    """
    try:
        directory = tempfile.mkdtemp(prefix=".flashhead-link-", dir=path)
    except OSError as exc:
        raise RuntimeError(f"Cannot write to {path}: {exc}.") from exc
    try:
        original = Path(directory) / "original"
        original.write_bytes(b"link check")
        try:
            os.link(original, Path(directory) / "link")
        except OSError:
            return False
        return True
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def detect_storage(workspace=WORKSPACE, alternate=GLOBAL_ALTERNATE):
    """Return ("network" | "global", data root) for the volume this pod has.

    FLASHHEAD_STORAGE=network or global overrides the detection. A Network Volume at
    /workspace wins over a Global Volume beside it, which keeps that layout unchanged.
    """
    workspace, alternate = Path(workspace), Path(alternate)
    forced = os.environ.get("FLASHHEAD_STORAGE", "auto").strip().lower() or "auto"
    if forced not in ("auto", "network", "global"):
        raise RuntimeError(f"FLASHHEAD_STORAGE={forced} is not auto, network or global.")
    workspace_kinds, alternate_kinds = mount_kinds(workspace), mount_kinds(alternate)
    if forced == "network":
        return "network", workspace
    if forced == "global":
        return "global", alternate if alternate_kinds else workspace
    if workspace_kinds and not any(k in EPHEMERAL for k in workspace_kinds):
        if any(object_filesystem(k) for k in workspace_kinds) or not hard_links_work(workspace):
            return "global", workspace
        if alternate_kinds:
            print(f"{LOG} Using the Network Volume at {workspace}; the Global Volume at "
                  f"{alternate} is not used. Set FLASHHEAD_STORAGE=global to use it instead.",
                  file=sys.stderr)
        return "network", workspace
    if alternate_kinds:
        return "global", alternate
    # Nothing usable: the Network Volume preflight explains what is missing.
    return "network", workspace


def require_global_volume(root):
    """Check a Global Volume for what this template does on it: plain file writes.

    ComfyUI, its virtualenv and custom nodes run from the container disk, so rename,
    hard links, locks, permission bits and execution are not needed on the volume.
    """
    root = Path(root).resolve()
    kinds = mount_kinds(root)
    if not kinds:
        raise RuntimeError(f"No volume mounted at {root}. Attach your Global Volume in the Pod storage settings.")
    if any(k in EPHEMERAL for k in kinds):
        raise RuntimeError(f"{root} uses {', '.join(kinds)}, not a persistent volume.")
    try:
        directory = tempfile.mkdtemp(prefix=".flashhead-write-", dir=root)
        try:
            probe = Path(directory) / "write-check"
            probe.write_bytes(b"global volume write check")
            if probe.read_bytes() != b"global volume write check":
                raise OSError("a file written to the volume read back differently")
        finally:
            shutil.rmtree(directory, ignore_errors=True)
    except OSError as exc:
        raise RuntimeError(f"Global Volume at {root} failed the write check: {exc}.") from exc
    print(f"{LOG} Global Volume ready at {root} ({kinds[-1]}): models, inputs, outputs and "
          "saved workflows stay there; ComfyUI runs from the container disk.")


def seed_missing(source, dest):
    """Copy files from source that dest does not have yet. Never overwrites."""
    source, dest = Path(source), Path(dest)
    for root, _, files in os.walk(source):
        target = dest / Path(root).relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            if not (target / name).exists():
                # copyfile, not copy2: the volume cannot take permission bits.
                shutil.copyfile(Path(root) / name, target / name)


def link_global_data(comfy, volume_comfy, folders=GLOBAL_DATA):
    """Point ComfyUI's data folders on the container disk at the Global Volume.

    The image's defaults (model folder layout, configs, examples) are copied over once,
    never over a file already on the volume; the local folder then becomes a link.
    """
    comfy, volume_comfy = Path(comfy), Path(volume_comfy)
    for folder in folders:
        local, remote = comfy / folder, volume_comfy / folder
        remote.mkdir(parents=True, exist_ok=True)
        if local.is_symlink():
            if Path(os.readlink(local)) == remote:
                continue
            local.unlink()
        elif local.is_dir():
            seed_missing(local, remote)
            shutil.rmtree(local)
        elif local.exists():
            raise RuntimeError(f"{local} is a file, not a folder; cannot link it to {remote}.")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(remote, target_is_directory=True)
        print(f"{LOG} {folder} -> {remote}")


def probe_working_volume(workspace):
    """Exercise required filesystem operations in a disposable private directory.

    This smoke check cannot prove atomicity or provider retention guarantees.
    Some network mounts expose fixed permission bits even when execution works.
    Test execution itself, rather than requiring an exact chmod result.
    """
    try:
        with tempfile.TemporaryDirectory(prefix=".flashhead-posix-", dir=workspace) as directory:
            base = Path(directory)
            original, replacement = base / "original", base / "replacement"
            original.write_bytes(b"old")
            replacement.write_bytes(b"new")
            os.replace(replacement, original)
            link = base / "link"
            link.symlink_to(original)
            if link.read_bytes() != b"new" or not link.is_symlink():
                raise OSError("symlink or file replacement did not work")
            executable = base / "execution-check"
            executable.write_bytes(b"#!/bin/sh\nexit 0\n")
            chmod_error = None
            try:
                executable.chmod(0o700)
            except OSError as exc:
                # A fixed-mode mount may reject chmod but already permit execution.
                chmod_error = exc
            try:
                # Execute directly: `bash script` would hide a noexec mount.
                subprocess.run([str(executable)], check=True, timeout=10,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (OSError, subprocess.SubprocessError) as exc:
                mode = executable.stat().st_mode & 0o777
                detail = f"; chmod failed: {chmod_error}" if chmod_error else ""
                raise OSError(f"could not execute a workspace file (mode {mode:#05o}{detail}): {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Working volume at {workspace} failed the POSIX check: {exc}. Use a regional Network Volume at /workspace.") from exc


def require_workspace_volume(workspace=Path("/workspace")):
    """Require a separate, non-ephemeral working mount before installing anything.

    Mount metadata cannot prove the provider's retention policy. Deployers must
    attach a regional Network Volume to retain the workspace after Pod deletion.
    Inspect mountinfo rather than ismount(), which can miss bind mounts.
    """
    workspace = Path(workspace).resolve()
    mountinfo = Path("/proc/self/mountinfo")
    instruction = "Attach a Network Volume or a Global Volume at /workspace in the Pod storage settings."
    if not mountinfo.exists():
        raise RuntimeError(f"Cannot verify the workspace mount. {instruction}")
    kinds = []
    for line in mountinfo.read_text().splitlines():
        fields = line.split()
        target = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
        if target == workspace:
            kinds.append(fields[fields.index("-") + 1].lower())
    if not kinds:
        raise RuntimeError(f"No separate volume mounted at {workspace}; refusing to install on the disposable container disk. {instruction}")
    if any(object_filesystem(kind) or any(name in kind for name in ("overlay", "tmpfs", "ramfs")) for kind in kinds):
        raise RuntimeError(f"Workspace mount uses {', '.join(kinds)}, not a persistent POSIX working volume. {instruction}")
    check_working_path(workspace)
    probe_working_volume(workspace)
    print(f"{LOG} Network Volume workspace ready at {workspace} ({kinds[-1]})")


def check_working_path(path):
    """Reject object-backed scratch, including a Global Volume at /workspace."""
    path = Path(path).resolve()
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.exists():
        matches = []
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            target = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
            if target == path or target in path.parents:
                matches.append((len(str(target)), fields[fields.index("-") + 1]))
        kind = max(matches, default=(0, ""))[1].lower()
        if object_filesystem(kind):
            raise RuntimeError(f"Working path {path} uses {kind}. Mount a Network Volume at /workspace for the working environment.")


def check_workspace_paths(comfy_dir, workspace=Path("/workspace")):
    """Reject old external data links instead of silently keeping a remote dependency."""
    workspace = Path(workspace).resolve()
    comfy = Path(comfy_dir)
    paths = [comfy, comfy / "models", comfy / "input", comfy / "output",
             comfy / "temp", comfy / "user", comfy / "user/default/workflows",
             Path(os.environ.get("HF_HOME", workspace / ".cache/huggingface")),
             workspace / ".jupyter", workspace / ".local/share/jupyter",
             workspace / ".ipython"]
    models = comfy / "models"
    if models.is_dir():
        paths.extend(p for p in models.rglob("*") if p.is_symlink())
    for path in paths:
        resolved = path.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise RuntimeError(
                f"Data path {path} points outside {workspace}: {resolved}. "
                "Copy existing data onto the Network Volume and replace the old "
                "link before restarting. No data has been deleted.")
        check_working_path(path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("detect")
    sub.add_parser("preflight").add_argument("path")
    sub.add_parser("preflight-global").add_argument("root")
    link = sub.add_parser("link-global")
    link.add_argument("comfy")
    link.add_argument("volume_comfy")
    args = parser.parse_args()
    if args.action == "detect":
        mode, root = detect_storage()
        print(mode, root)
    elif args.action == "preflight":
        require_workspace_volume()
        check_workspace_paths(args.path)
    elif args.action == "preflight-global":
        require_global_volume(args.root)
    else:
        link_global_data(args.comfy, args.volume_comfy)


if __name__ == "__main__":
    main()
