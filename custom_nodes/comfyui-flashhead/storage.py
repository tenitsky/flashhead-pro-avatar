"""Preflight checks for the persistent Network Volume workspace."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

LOG = "[FlashHead Storage]"


def object_filesystem(kind):
    """FUSE is an interface, not a storage type; POSIX network mounts use it too."""
    return any(name in kind.lower() for name in ("s3fs", "geesefs", "gcsfuse", "rclone", "fuse.global"))


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
    instruction = "Attach your Network Volume at /workspace in the Pod storage settings."
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
    parser.add_argument("action", choices=["preflight"])
    parser.add_argument("path")
    args = parser.parse_args()
    require_workspace_volume()
    check_workspace_paths(args.path)


if __name__ == "__main__":
    main()
