"""Run the complete setup script against a small, offline Linux image fixture.

Filesystem operations and setup control flow are real. Mount metadata, fixed chmod
behaviour, git, pip, downloads, the GPU and the image service handoff are simulated;
this does not replace booting the actual image on RunPod.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
sys.path.insert(0, str(ROOT / "tools"))
import build_workflow as builder  # noqa: E402

# Every downloaded file, relative to ComfyUI/models/flashhead.
MODEL_PATHS = {f"{folder}/{path}" for path, folder, _ in builder.MODEL_FILES}


@unittest.skipUnless(sys.platform.startswith("linux") and BASH, "Linux Bash integration fixture")
class BootTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="flashhead-pro-boot-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        # A Global Volume mounted beside /workspace, and the container-disk install
        # setup uses with a Global Volume.
        self.alternate = self.base / "workspace-global"
        self.local = self.base / "container-disk"
        self.local_comfy = self.local / "ComfyUI"
        self.comfy = self.workspace / "runpod-slim/ComfyUI"
        self.upstream = self.comfy / "custom_nodes/comfyui-flashhead/SoulX-FlashHead"
        self.baked = self.base / "baked"
        (self.baked / "comfy_api/latest").mkdir(parents=True)
        (self.baked / "main.py").write_text("# image fixture, no GPU service\n", encoding="utf-8")
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.site = self.base / "site"
        self.site.mkdir()
        self.handoff = self.base / "handoff.json"
        self.download_log = self.base / "downloads.log"
        self.pip_log = self.base / "pip.log"
        self.git_log = self.base / "git.log"
        self.apt_log = self.base / "apt.log"
        self.start = self.base / "start.sh"
        self.constraints = self.base / "constraints.txt"
        self.constraints.write_text("torch==2.10.0\n", encoding="utf-8")

        # Keep the shipped code, nodes and workflows. Only container paths are
        # rewritten; no installation function or branch is replaced.
        shutil.copytree(ROOT / "custom_nodes", self.repo / "custom_nodes")
        shutil.copytree(ROOT / "workflows", self.repo / "workflows")
        shutil.copytree(ROOT / "samples", self.repo / "samples")
        storage = self.repo / "custom_nodes/comfyui-flashhead/storage.py"
        storage.write_text(storage.read_text(encoding="utf-8").replace("/workspace", str(self.workspace)), encoding="utf-8")
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        # The temp-file rewrite goes first: the fixture's own base directory is
        # under /tmp/flashhead-boot-*, and must not be rewritten again.
        for original, replacement in (
            ("/tmp/flashhead-", str(self.base / "tmp-flashhead-")),
            ("/workspace", str(self.workspace)),
            ("/start.sh", str(self.start)),
            ("/opt/comfyui-baked", str(self.baked)),
            ("/opt/flashhead-local", str(self.local)),
            ("/opt/comfyui-runtime-constraints.txt", str(self.constraints)),
        ):
            setup = setup.replace(original, replacement)
        self.setup_script = self.repo / "setup.sh"
        self.setup_script.write_text(setup, encoding="utf-8", newline="\n")

        self.fixture = self.base / "fixture.safetensors"
        header = json.dumps({"tensor": {"dtype": "U8", "shape": [1000000], "data_offsets": [0, 1000000]}}).encode()
        header += b" " * (-len(header) % 8)
        self.fixture.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 1000000)

        # Shaped like torch.save output: a zip holding archive/data.pkl, over 1 MB.
        self.checkpoint = self.base / "fixture.pth"
        import zipfile
        with zipfile.ZipFile(self.checkpoint, "w") as z:
            z.writestr("archive/data.pkl", b"fixture")
            z.writestr("archive/data/0", b"\0" * 1000000)

        real = shlex.quote(sys.executable)
        # ComfyUI's venv: `-m pip` is recorded, everything else is the real
        # interpreter (with the fixture site packages, as --system-site-packages).
        venv_python = ("#!/bin/bash\n"
                       'if [ "${1-}" = -m ] && [ "${2-}" = pip ]; then\n'
                       '  printf \'%s\\n\' "$*" >> "$BOOT_PIP_LOG"\n'
                       '  exit "${BOOT_PIP_EXIT:-0}"\n'
                       "fi\n"
                       "exec " + real + ' "$@"\n')
        python = ("#!/bin/bash\n"
                  'if [ "${1-}" = -m ] && [ "${2-}" = venv ]; then\n'
                  '  case " $* " in\n'
                  '    *" --system-site-packages "*)\n'
                  '      dir="${@: -1}"; mkdir -p "$dir/bin"\n'
                  "      cat > \"$dir/bin/python\" <<'VENV'\n" + venv_python + "VENV\n"
                  '      chmod +x "$dir/bin/python"; exit 0 ;;\n'
                  "  esac\n"
                  "  exit 1  # the optional downloader venv: exercise the wget fallback\n"
                  "fi\n"
                  "exec " + real + ' "$@"\n')
        self.make_executable(self.bin / "python3", python)
        self.make_executable(self.bin / "python3.12", python)
        self.make_executable(self.bin / "apt-get", """#!/bin/bash
printf '%s\\n' "$*" >> "$BOOT_APT_LOG"
exit 0
""")
        self.make_executable(self.bin / "ffmpeg", "#!/bin/bash\nexit 0\n")
        self.make_executable(self.bin / "sleep", "#!/bin/bash\nexit 0\n")
        # git: the template revision lookup fails (no checkout); the upstream checkout is a
        # fake repository whose checkout writes the files setup checks for.
        self.make_executable(self.bin / "git", """#!/bin/bash
printf '%s\\n' "$*" >> "$BOOT_GIT_LOG"
if [ "$1" = init ]; then mkdir -p "${@: -1}/.git"; exit 0; fi
[ "$1" = -C ] || exit 1
dir="$2"; shift 2
[ "$1" = -c ] && shift 2
case "$1" in
  rev-parse)
    [ -f "$dir/.git/HEAD_SHA" ] || exit 1
    sha=$(cat "$dir/.git/HEAD_SHA")
    if [ "$2" = --short ]; then echo "${sha:0:7}"; else echo "$sha"; fi ;;
  remote) exit 0 ;;
  status) [ -n "${BOOT_GIT_DIRTY:-}" ] && echo " M nodes.py"; exit 0 ;;
  fetch) [ -z "${BOOT_GIT_OFFLINE:-}" ] || exit 1; echo "${@: -1}" > "$dir/.git/FETCH_HEAD" ;;
  checkout)
    cp "$dir/.git/FETCH_HEAD" "$dir/.git/HEAD_SHA"
    mkdir -p "$dir/flash_head/src/pipeline"
    echo "# fixture" > "$dir/flash_head/src/pipeline/flash_head_pipeline.py" ;;
  *) exit 1 ;;
esac
""")
        self.make_executable(self.bin / "wget", """#!/bin/bash
set -e
destination=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = -O ]; then
    shift
    destination="$1"
  fi
  shift
done
test -n "$destination"
printf '%s\\n' "$destination" >> "$BOOT_DOWNLOAD_LOG"
case "$destination" in
  *.json.part) printf '{"fixture": true}' > "$destination" ;;
  *.pth.part) cp "$BOOT_CHECKPOINT_FIXTURE" "$destination" ;;
  *) cp "$BOOT_MODEL_FIXTURE" "$destination" ;;
esac
if [ -n "${BOOT_FAIL_MODEL:-}" ] && [[ "$destination" = *"$BOOT_FAIL_MODEL"* ]]; then
  exit 1  # A full-sized partial must not be promoted after a failed transfer.
fi
""")
        # A Global Volume has no rename, hard links, symlinks or permission bits. The
        # fixture refuses all four on it, in Python and for mv/ln, so a setup step that
        # relied on one fails here instead of on RunPod.
        real_mv, real_ln = shutil.which("mv"), shutil.which("ln")
        self.make_executable(self.bin / "mv", """#!/bin/bash
if [ -n "${BOOT_OBJECT_ROOT:-}" ]; then
  for arg in "$@"; do
    case "$(realpath -m -- "$arg")" in
      "$BOOT_OBJECT_ROOT"|"$BOOT_OBJECT_ROOT"/*) echo "mv: no rename on the Global Volume: $arg" >&2; exit 1 ;;
    esac
  done
fi
exec """ + shlex.quote(real_mv) + ' "$@"\n')
        self.make_executable(self.bin / "ln", """#!/bin/bash
if [ -n "${BOOT_OBJECT_ROOT:-}" ]; then
  case "$(realpath -m -s -- "${@: -1}")" in
    "$BOOT_OBJECT_ROOT"|"$BOOT_OBJECT_ROOT"/*) echo "ln: no links on the Global Volume: ${@: -1}" >&2; exit 1 ;;
  esac
fi
exec """ + shlex.quote(real_ln) + ' "$@"\n')
        self.make_executable(self.start, f"""#!/bin/bash
set -e
COMFYUI_DIR="{self.workspace}/runpod-slim/ComfyUI"
DB_FILE="{self.workspace}/runpod-slim/filebrowser.db"
start_jupyter() {{
  if [ "${{JUPYTER_DISABLE_AUTH:-}}" = "true" ]; then JUPYTER_TOKEN=""; fi
}}
export START_SCRIPT="$0" START_COMFYUI_DIR="$COMFYUI_DIR" START_DB_FILE="$DB_FILE"
# The image service boundary is replaced, not the template's setup script.
python3 - <<'PY'
import json, os
from pathlib import Path
keys = ('COMFYUI_PATH', 'HF_HOME', 'JUPYTER_CONFIG_DIR', 'JUPYTER_DATA_DIR',
        'JUPYTER_RUNTIME_DIR', 'IPYTHONDIR', 'JUPYTER_DISABLE_AUTH', 'START_SCRIPT',
        'START_COMFYUI_DIR', 'START_DB_FILE', 'FB_DATABASE')
Path(os.environ['BOOT_HANDOFF']).write_text(json.dumps({{k: os.environ.get(k) for k in keys}}))
PY
""")

        (self.site / "sitecustomize.py").write_text("""import os
from pathlib import Path

import errno

workspace = Path(os.environ['BOOT_WORKSPACE']).resolve()
alternate = Path(str(workspace) + '-global')
# network: a Network Volume at /workspace. global: a Global Volume there.
# global-alternate: a Global Volume at /workspace-global, nothing at /workspace.
mounts = os.environ.get('BOOT_MOUNTS', 'network')
object_root = {'global': workspace, 'global-alternate': alternate}.get(mounts)
read_text = Path.read_text
chmod, link, symlink, rename, replace = os.chmod, os.link, os.symlink, os.rename, os.replace

def fixture_read_text(path, *args, **kwargs):
    if path == Path('/proc/self/mountinfo'):
        text = '1 0 0:1 / / rw - overlay overlay rw\\n'
        if os.environ.get('BOOT_MISSING_VOLUME') != '1':
            if mounts == 'network':
                text += f'2 1 0:2 / {workspace} rw - fuse.vstorage volume rw\\n'
            else:
                text += f'2 1 0:2 / {object_root} rw - fuse volume rw\\n'
        return text
    return read_text(path, *args, **kwargs)

def on_object_volume(path, follow=True):
    if object_root is None or isinstance(path, int):
        return False
    path = Path(os.fsdecode(path))
    path = path.resolve() if follow else path.parent.resolve() / path.name
    return path == object_root or object_root in path.parents

def unsupported(what):
    raise OSError(errno.EPERM, f'Global Volume fixture: no {what}')

def fixed_workspace_chmod(path, mode, *args, **kwargs):
    if on_object_volume(path):
        unsupported('permission bits')
    if not isinstance(path, int):
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved == workspace or workspace in resolved.parents:
            mode = 0o777
    return chmod(path, mode, *args, **kwargs)

def object_link(src, dst, *args, **kwargs):
    if on_object_volume(src) or on_object_volume(dst, follow=False):
        unsupported('hard links')
    return link(src, dst, *args, **kwargs)

def object_symlink(src, dst, *args, **kwargs):
    if on_object_volume(dst, follow=False):
        unsupported('symlinks')
    return symlink(src, dst, *args, **kwargs)

def object_rename(function):
    def wrapped(src, dst, *args, **kwargs):
        if on_object_volume(src, follow=False) or on_object_volume(dst, follow=False):
            unsupported('rename')
        return function(src, dst, *args, **kwargs)
    return wrapped

Path.read_text = fixture_read_text
os.chmod = fixed_workspace_chmod
os.link, os.symlink = object_link, object_symlink
os.rename, os.replace = object_rename(rename), object_rename(replace)
""", encoding="utf-8", newline="\n")
        # The image's PyTorch, as setup's GPU check sees it, and the packages the
        # dependency check imports.
        (self.site / "torch").mkdir()
        (self.site / "torch/__init__.py").write_text("""import os
from types import SimpleNamespace

__version__ = "2.10.0+fixture"
version = SimpleNamespace(cuda=os.environ.get("BOOT_TORCH_CUDA", "12.8") or None)
cuda = SimpleNamespace(is_available=lambda: os.environ.get("BOOT_GPU", "1") == "1",
                       get_device_name=lambda index: "Fixture GPU")
""", encoding="utf-8", newline="\n")
        packages = {name: "" for name in ("loguru", "pyloudnorm", "einops", "safetensors",
                                          "transformers", "torchaudio", "PIL")}
        packages["diffusers"] = '__version__ = "0.36.0"\n'
        packages["cv2"] = '__version__ = "5.0.0"\nFaceDetectorYN = object\n'
        for name, source in packages.items():
            (self.site / name).mkdir()
            (self.site / name / "__init__.py").write_text(source, encoding="utf-8")

        self.env = dict(os.environ)
        for key in ("COMFYUI_PATH", "HF_HOME", "HF_TOKEN", "JUPYTER_NO_AUTH",
                    "FLASHHEAD_REF", "COMFYUI_VENV_NAME", "FLASHHEAD_STORAGE"):
            self.env.pop(key, None)
        self.env.update({
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(self.site),
            "PYTHONDONTWRITEBYTECODE": "1",
            "BOOT_WORKSPACE": str(self.workspace),
            "BOOT_HANDOFF": str(self.handoff),
            "BOOT_DOWNLOAD_LOG": str(self.download_log),
            "BOOT_PIP_LOG": str(self.pip_log),
            "BOOT_GIT_LOG": str(self.git_log),
            "BOOT_APT_LOG": str(self.apt_log),
            "BOOT_MODEL_FIXTURE": str(self.fixture),
            "BOOT_CHECKPOINT_FIXTURE": str(self.checkpoint),
        })

    @staticmethod
    def make_executable(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def boot(self, **overrides):
        self.handoff.unlink(missing_ok=True)
        return subprocess.run([BASH, str(self.setup_script)], env={**self.env, **overrides},
                              text=True, capture_output=True, timeout=60)

    def lines(self, path):
        return path.read_text().splitlines() if path.exists() else []

    def boot_global(self, alternate=False, **overrides):
        root = self.alternate if alternate else self.workspace
        if alternate:
            self.alternate.mkdir(exist_ok=True)
        return self.boot(BOOT_MOUNTS="global-alternate" if alternate else "global",
                         BOOT_OBJECT_ROOT=str(root.resolve()), **overrides)

    def assert_global_boot(self, result, volume_root):
        """A completed Global Volume boot: data on the volume, software on the disk."""
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Storage: global volume at {volume_root}", result.stdout)
        volume = volume_root / "runpod-slim/ComfyUI"
        models = volume / "models/flashhead"
        self.assertEqual({p.relative_to(models).as_posix() for p in models.rglob("*")
                          if p.is_file()}, MODEL_PATHS)
        for w in builder.WORKFLOWS:
            self.assertTrue((volume / "user/default/workflows" / w["path"]).is_file(), w["path"])
        for sample in (ROOT / "samples").iterdir():
            self.assertTrue((volume / "input" / sample.name).is_file(), sample.name)
        self.assertTrue((volume / "input/batch_audio").is_dir())
        for folder in ("models", "input", "output", "user/default"):
            link = self.local_comfy / folder
            self.assertTrue(link.is_symlink(), folder)
            self.assertEqual(link.resolve(), (volume / folder).resolve())
        # ComfyUI, its venv, the node pack and upstream's code never touch the volume.
        self.assertTrue((self.local_comfy / "main.py").is_file())
        self.assertTrue((self.local_comfy / ".venv-cu128/.flashhead-pro-avatar-deps").is_file())
        upstream = self.local_comfy / "custom_nodes/comfyui-flashhead/SoulX-FlashHead"
        self.assertEqual((upstream / ".git/HEAD_SHA").read_text().strip(), builder.FLASHHEAD_REF)
        self.assertEqual(sorted(p.name for p in volume.iterdir()),
                         ["input", "models", "output", "user"])
        self.assertFalse(list(volume_root.rglob("*.part")))
        self.assertFalse(list(volume_root.rglob(".flashhead-*")))
        # Every download was staged on the container disk, which is empty afterwards.
        stage = self.local / "downloads"
        for destination in self.lines(self.download_log):
            self.assertIn(stage, Path(destination).parents)
        self.assertEqual(list(stage.iterdir()), [])
        state = json.loads(self.handoff.read_text())
        self.assertEqual(state["START_SCRIPT"], str(self.local / "image-start.sh"))
        self.assertEqual(state["START_COMFYUI_DIR"], str(self.local_comfy))
        self.assertEqual(state["COMFYUI_PATH"], str(self.local_comfy))
        self.assertEqual(state["FB_DATABASE"], str(self.local / "filebrowser.db"))
        for key in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "IPYTHONDIR", "HF_HOME"):
            self.assertIn(self.local, Path(state[key]).parents, key)
        # The image's own start script is run as a copy, never edited.
        self.assertIn(f'COMFYUI_DIR="{self.workspace}/runpod-slim/ComfyUI"', self.start.read_text())
        return volume

    def test_global_volume_first_boot_and_fresh_container(self):
        first = self.boot_global()
        volume = self.assert_global_boot(first, self.workspace)
        self.assertIn("Global Volume ready", first.stdout)
        self.assertEqual(len(self.lines(self.download_log)), len(MODEL_PATHS))
        self.assertEqual(len(self.lines(self.pip_log)), 1)
        for sample in (ROOT / "samples").iterdir():
            self.assertEqual((volume / "input" / sample.name).read_bytes(), sample.read_bytes())
        edited = volume / "user/default/workflows" / builder.WORKFLOWS[0]["path"]
        edited.write_text('{"edited_by_user": true}\n')
        render = volume / "output/my_render.mp4"
        render.write_bytes(b"video")
        # Restart with the container disk kept: everything is reused.
        self.assert_global_boot(self.boot_global(), self.workspace)
        self.assertEqual(len(self.lines(self.download_log)), len(MODEL_PATHS))
        self.assertEqual(len(self.lines(self.pip_log)), 1)
        # A fresh container disk: ComfyUI and its packages are set up again, the models
        # and user files are reused.
        shutil.rmtree(self.local)
        self.assert_global_boot(self.boot_global(), self.workspace)
        self.assertEqual(len(self.lines(self.download_log)), len(MODEL_PATHS))
        self.assertEqual(len(self.lines(self.pip_log)), 2)
        self.assertEqual(edited.read_text(), '{"edited_by_user": true}\n')
        self.assertEqual(render.read_bytes(), b"video")

    def test_global_volume_reuses_models_copied_from_a_network_volume(self):
        # RunPod's migration copies the whole Network Volume over with rsync.
        self.assertEqual(self.boot().returncode, 0)
        result = self.boot_global()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.lines(self.download_log)), len(MODEL_PATHS),
                         "migrated models must be reused")
        self.assertEqual(json.loads(self.handoff.read_text())["START_COMFYUI_DIR"],
                         str(self.local_comfy))

    def test_global_volume_beside_an_empty_workspace(self):
        self.assert_global_boot(self.boot_global(alternate=True), self.alternate)
        self.assertEqual((self.workspace / "global-volume").resolve(), self.alternate.resolve())

    def test_global_volume_failed_download_never_hands_off(self):
        result = self.boot_global(BOOT_FAIL_MODEL="diffusion_pytorch_model.safetensors")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FATAL: one or more model downloads failed", result.stdout)
        self.assertFalse(self.handoff.exists())
        self.assertFalse(list(self.workspace.rglob("diffusion_pytorch_model.safetensors*")))
        # The partial stays on the container disk for wget to resume.
        self.assertTrue(list((self.local / "downloads").glob("*diffusion_pytorch_model.safetensors.part")))

    def test_global_volume_with_an_unknown_start_script_stops_before_downloads(self):
        self.start.write_text("#!/bin/bash\necho different image\n")
        result = self.boot_global()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not set COMFYUI_DIR", result.stdout)
        self.assertFalse(self.handoff.exists())
        self.assertEqual(self.lines(self.download_log), [])

    def test_network_mode_forced_on_a_global_volume_fails_its_posix_check(self):
        result = self.boot_global(FLASHHEAD_STORAGE="network")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed the POSIX check", result.stderr)
        self.assertFalse(self.comfy.exists())
        self.assertFalse(self.handoff.exists())

    def test_first_boot_and_restart_preserve_user_files_and_reuse_everything(self):
        first = self.boot()
        log = first.stdout + first.stderr
        self.assertEqual(first.returncode, 0, log)
        self.assertTrue(self.handoff.exists(), log)
        self.assertIn("PyTorch 2.10.0+fixture, CUDA build 12.8", first.stdout)
        self.assertIn("GPU: Fixture GPU", first.stdout)
        # Models: seven files, in the repo layout upstream loads.
        models = self.comfy / "models/flashhead"
        self.assertEqual(len(self.lines(self.download_log)), len(MODEL_PATHS))
        self.assertEqual({p.relative_to(models).as_posix() for p in models.rglob("*")
                          if p.is_file()}, MODEL_PATHS)
        for config in models.rglob("*.json"):
            json.loads(config.read_text())
        # Nodes: the pack's files and upstream's code at the pinned commit.
        pack = self.comfy / "custom_nodes/comfyui-flashhead"
        for name in ("__init__.py", "engine.py", "storage.py",
                     "face_detection_yunet_2023mar.onnx"):
            self.assertTrue((pack / name).is_file(), name)
        self.assertEqual((self.upstream / ".git/HEAD_SHA").read_text().strip(),
                         builder.FLASHHEAD_REF)
        self.assertIn(f"SoulX-FlashHead: {builder.FLASHHEAD_REF[:7]}", first.stdout)
        # Packages: into ComfyUI's venv, pinned to the image's torch, no upstream
        # requirements.txt (it would pull gradio, xformers, xfuser and mediapipe).
        venv = self.comfy / ".venv-cu128"
        pip = self.lines(self.pip_log)
        self.assertEqual(len(pip), 1, pip)
        self.assertIn(f"-c {self.constraints}", pip[0])
        self.assertIn("diffusers==0.36.0", pip[0])
        self.assertNotIn("requirements.txt", pip[0])
        self.assertTrue((venv / ".flashhead-pro-avatar-deps").is_file())
        # Workflows: all of them in the sidebar folder.
        sidebar = self.comfy / "user/default/workflows"
        for w in builder.WORKFLOWS:
            self.assertEqual(json.loads((sidebar / w["path"]).read_text()),
                             json.loads((ROOT / "workflows" / w["path"]).read_text()))
        self.assertTrue((self.comfy / "input/batch_audio").is_dir())
        # The samples the workflows open on, copied into ComfyUI's input folder.
        for sample in (ROOT / "samples").iterdir():
            self.assertEqual((self.comfy / "input" / sample.name).read_bytes(),
                             sample.read_bytes())

        state = json.loads(self.handoff.read_text())
        self.assertEqual(state["COMFYUI_PATH"], str(self.comfy))
        self.assertEqual(state["JUPYTER_DISABLE_AUTH"], "true")
        runtime = Path(state["JUPYTER_RUNTIME_DIR"])
        self.assertNotIn(self.workspace, runtime.parents)
        self.assertEqual(runtime.stat().st_mode & 0o777, 0o700)
        for key in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "IPYTHONDIR", "HF_HOME"):
            self.assertIn(self.workspace, Path(state[key]).parents)

        edited = sidebar / builder.WORKFLOWS[0]["path"]
        edited.write_text('{"edited_by_user": true}\n')
        # A user's own file under a sample's name is never overwritten.
        own_portrait = self.comfy / "input" / builder.SAMPLE_PORTRAIT
        own_portrait.write_bytes(b"my portrait")
        notebook = self.workspace / "my-notebook.ipynb"
        notebook.write_text('{"cells": []}\n')
        git_calls = len(self.lines(self.git_log))

        restart = self.boot()
        self.assertEqual(restart.returncode, 0, restart.stdout + restart.stderr)
        self.assertTrue(self.handoff.exists())
        self.assertEqual(len(self.lines(self.download_log)), len(MODEL_PATHS),
                         "models must not download again")
        self.assertEqual(len(self.lines(self.pip_log)), 1, "packages must not reinstall")
        self.assertIn("already installed", restart.stdout)
        new_git = self.lines(self.git_log)[git_calls:]
        self.assertFalse([c for c in new_git if " fetch " in f" {c} "], new_git)
        self.assertEqual(edited.read_text(), '{"edited_by_user": true}\n')
        self.assertEqual(notebook.read_text(), '{"cells": []}\n')
        self.assertEqual(own_portrait.read_bytes(), b"my portrait")
        self.assertEqual(list(self.workspace.glob(".flashhead-posix-*")), [])

    def test_new_pin_moves_the_upstream_checkout(self):
        self.assertEqual(self.boot().returncode, 0)
        result = self.boot(FLASHHEAD_REF="0123456789abcdef0123456789abcdef01234567")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.upstream / ".git/HEAD_SHA").read_text().strip(),
                         "0123456789abcdef0123456789abcdef01234567")

    def test_upstream_with_local_changes_is_not_overwritten(self):
        self.assertEqual(self.boot().returncode, 0)
        result = self.boot(FLASHHEAD_REF="0123456789abcdef0123456789abcdef01234567",
                           BOOT_GIT_DIRTY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has local changes; refusing to overwrite", result.stdout)
        self.assertEqual((self.upstream / ".git/HEAD_SHA").read_text().strip(), builder.FLASHHEAD_REF)
        self.assertFalse(self.handoff.exists())

    def test_upstream_fetch_failure_stops_with_a_reason(self):
        result = self.boot(BOOT_GIT_OFFLINE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SoulX-FlashHead is not at", result.stdout)
        self.assertEqual(self.lines(self.download_log), [])
        self.assertFalse(self.handoff.exists())

    def test_pip_failure_stops_before_downloads(self):
        result = self.boot(BOOT_PIP_EXIT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installing FlashHead's Python packages failed", result.stdout)
        self.assertEqual(self.lines(self.download_log), [])
        self.assertFalse((self.comfy / ".venv-cu128/.flashhead-pro-avatar-deps").exists())
        self.assertFalse(self.handoff.exists())

    def test_missing_volume_stops_before_installation(self):
        result = self.boot(BOOT_MISSING_VOLUME="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No separate volume mounted", result.stderr)
        self.assertFalse(self.comfy.exists())
        self.assertFalse(self.handoff.exists())
        self.assertEqual(self.lines(self.download_log), [])
        self.assertFalse(self.apt_log.exists())

    def test_unusable_gpu_stops_before_downloads(self):
        result = self.boot(BOOT_GPU="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("host driver is too old", result.stderr)
        self.assertEqual(self.lines(self.download_log), [])
        self.assertEqual(self.lines(self.pip_log), [])
        self.assertFalse(self.handoff.exists())

    def test_failed_model_transfer_never_hands_off_to_services(self):
        failed = "Model_Pro/diffusion_pytorch_model.safetensors"
        result = self.boot(BOOT_FAIL_MODEL=failed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FATAL: one or more model downloads failed", result.stdout)
        self.assertFalse(self.handoff.exists())
        models = self.comfy / "models/flashhead/SoulX-FlashHead-1_3B"
        self.assertFalse((models / failed).exists())
        self.assertTrue((models / (failed + ".part")).exists())
        # The other files still finished, so a restart only fetches the failed one.
        done = {p.relative_to(self.comfy / "models/flashhead").as_posix()
                for p in (self.comfy / "models/flashhead").rglob("*") if p.suffix != ".part"
                and p.is_file()}
        self.assertEqual(done, MODEL_PATHS - {"SoulX-FlashHead-1_3B/" + failed})

    def test_existing_incomplete_comfy_directory_is_not_deleted(self):
        self.comfy.mkdir(parents=True)
        saved_file = self.comfy / "keep-my-data.txt"
        saved_file.write_text("existing user data")
        result = self.boot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to delete user files", result.stdout)
        self.assertEqual(saved_file.read_text(), "existing user data")
        self.assertFalse((self.comfy / "main.py").exists())
        self.assertFalse(self.handoff.exists())
        self.assertEqual(self.lines(self.download_log), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
