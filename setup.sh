#!/bin/bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting FlashHead Pro Avatar Template Setup ==="
echo "Template revision: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || printf unknown)"

# Refuse a disposable workspace before doing installation work.
# The image's /start.sh fixes ComfyUI at this path.
export COMFYUI_PATH="${COMFYUI_PATH:-/workspace/runpod-slim/ComfyUI}"
if [ "$COMFYUI_PATH" != /workspace/runpod-slim/ComfyUI ]; then
  echo "FATAL: this image's /start.sh uses /workspace/runpod-slim/ComfyUI. Leave COMFYUI_PATH unset."
  exit 1
fi
python3 "$SCRIPT_DIR/custom_nodes/comfyui-flashhead/storage.py" preflight "$COMFYUI_PATH"

# Apply before the image starts Jupyter. The image's /start.sh runs JupyterLab
# without a token when JUPYTER_DISABLE_AUTH=true, uses JUPYTER_PASSWORD as the
# token when set, and otherwise does not start Jupyter at all. The exported value
# reaches /start.sh through the final exec. Jupyter is optional, so a start script
# without that switch is reported rather than stopping the pod.
if [ "${JUPYTER_NO_AUTH:-1}" = "1" ]; then
  export JUPYTER_DISABLE_AUTH=true
  echo "Jupyter authentication disabled: anyone with access to its URL can run commands."
  if ! grep -q 'JUPYTER_DISABLE_AUTH' /start.sh 2>/dev/null; then
    echo "WARNING: this image's /start.sh has no JUPYTER_DISABLE_AUTH switch, so JupyterLab"
    echo "         may not start. Use runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8, or set"
    echo "         JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD. ComfyUI is unaffected."
  fi
else
  unset JUPYTER_DISABLE_AUTH
  if [ -z "${JUPYTER_PASSWORD:-}" ]; then
    echo "NOTE: JUPYTER_NO_AUTH=0 without JUPYTER_PASSWORD: the image does not start JupyterLab."
  fi
fi

echo "=== Ensuring System Dependencies are Installed ==="
# The image already supplies these. Contact apt mirrors only if something is
# actually missing; an unrelated mirror outage must not break a complete image.
packages=()
command -v wget >/dev/null || packages+=(wget)
command -v git >/dev/null || packages+=(git)
command -v ffmpeg >/dev/null || packages+=(ffmpeg)
[ -s /etc/ssl/certs/ca-certificates.crt ] || packages+=(ca-certificates)
if [ "${#packages[@]}" -gt 0 ]; then
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

# Jupyter serves /workspace in the base image. Keep its settings and user data on
# that persistent volume too; /root belongs to the disposable container disk.
export JUPYTER_CONFIG_DIR=/workspace/.jupyter
export JUPYTER_DATA_DIR=/workspace/.local/share/jupyter
export IPYTHONDIR=/workspace/.ipython
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$IPYTHONDIR"

# Network Volumes can expose fixed permissions. Jupyter's cookie secrets and
# kernel connection files require private 0600 files; these are disposable
# runtime state, not notebooks or settings. Keep them on the container filesystem.
JUPYTER_RUNTIME_DIR="$(mktemp -d /tmp/flashhead-pro-jupyter-runtime.XXXXXX)"
export JUPYTER_RUNTIME_DIR
chmod 700 "$JUPYTER_RUNTIME_DIR"
python3 - <<'PY'
import os
from pathlib import Path
from jupyter_core.paths import secure_write

probe = Path(os.environ["JUPYTER_RUNTIME_DIR"]) / "secure-write-check"
try:
    with secure_write(str(probe)) as stream:
        stream.write("Jupyter runtime permissions verified")
finally:
    probe.unlink(missing_ok=True)
print(f"Jupyter runtime ready at {probe.parent}; notebooks and settings remain on /workspace.")
PY

# SoulX-FlashHead has no ComfyUI nodes of its own. This repo's comfyui-flashhead pack
# drives upstream's pipeline code, pinned here to the commit the template was built and
# tested against. Override with FLASHHEAD_REF only to test a newer commit.
#   Base image: runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8
# This script deliberately does NOT update ComfyUI core: on these images
# `pip install -r requirements.txt` can reinstall torch and break the baked CUDA build.
FLASHHEAD_URL="https://github.com/Soul-AILab/SoulX-FlashHead.git"
FLASHHEAD_REF="${FLASHHEAD_REF:-9bc03de06bb0de82cd6bc477804512ae06144bf2}"

# Both model repos are ungated, so no token is needed. Set HF_TOKEN anyway if you have
# one: anonymous requests are rate-limited (~120/h vs ~1000/h) and slower.
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN || echo "NOTE: HF_TOKEN not set - downloads may be slower and rate-limited."
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  if [ -e "$COMFYUI_PATH" ]; then
    echo "FATAL: existing ComfyUI directory has no main.py; refusing to delete user files."
    exit 1
  fi
  mkdir -p "$(dirname "$COMFYUI_PATH")"
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "FATAL: no ComfyUI at $COMFYUI_PATH and /opt/comfyui-baked did not provide one."
  echo "       Check the container image."
  exit 1
fi
echo "ComfyUI: $COMFYUI_PATH"

# Validation only needs the standard library. Use the image interpreter that
# already passed preflight, not a possibly stale persisted ComfyUI virtualenv.
PY="$(command -v python3)"
echo "Setup validator Python: $PY"

# PyTorch comes from the image (the volume's venv only adds custom-node packages).
# The image also starts on hosts whose driver is too old for its CUDA build, where
# PyTorch cannot see the GPU - stop here, before the downloads, rather than render on
# the CPU.
TORCH_PY="$(command -v python3.12 || command -v python3)"
if ! "$TORCH_PY" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"FATAL: cannot import the image's PyTorch ({exc}).")
build = torch.version.cuda or "none"
print(f"PyTorch {torch.__version__}, CUDA build {build}")
if build == "none":
    sys.exit("FATAL: the image's PyTorch has no CUDA build. Change the template's "
             "container image to runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8.")
if not torch.cuda.is_available():
    sys.exit(f"FATAL: CUDA {build} PyTorch cannot use this GPU: the host driver is too "
             "old for it. Redeploy on another GPU or datacenter; the same Network "
             "Volume can be reattached.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
then
  exit 1
fi

# 1. Custom nodes
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
NODE_DIR="$COMFYUI_PATH/custom_nodes/comfyui-flashhead"
mkdir -p "$NODE_DIR"
# The pack's own files; the upstream checkout inside it is managed below.
cp "$SCRIPT_DIR/custom_nodes/comfyui-flashhead/"*.py \
   "$SCRIPT_DIR/custom_nodes/comfyui-flashhead/"*.onnx "$NODE_DIR/"

UPSTREAM_DIR="$NODE_DIR/SoulX-FlashHead"
upstream_at_ref() {
  [ "$(git -C "$UPSTREAM_DIR" rev-parse HEAD 2>/dev/null)" = "$FLASHHEAD_REF" ]
}
if [ -d "$UPSTREAM_DIR" ] && [ ! -d "$UPSTREAM_DIR/.git" ]; then
  echo "FATAL: $UPSTREAM_DIR exists but is not a git checkout; refusing to replace it."
  echo "       Move it aside and restart to install the pinned version."
  exit 1
fi
if ! upstream_at_ref; then
  echo "Installing SoulX-FlashHead at ${FLASHHEAD_REF:0:12}..."
  if [ ! -d "$UPSTREAM_DIR" ]; then
    git init -q "$UPSTREAM_DIR"
    git -C "$UPSTREAM_DIR" remote add origin "$FLASHHEAD_URL"
  elif [ -n "$(git -C "$UPSTREAM_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "FATAL: $UPSTREAM_DIR has local changes; refusing to overwrite them."
    echo "       Commit or discard them, or set FLASHHEAD_REF to its current commit."
    exit 1
  fi
  fetched=0
  for attempt in 1 2 3; do
    if git -C "$UPSTREAM_DIR" fetch -q --depth 1 "$FLASHHEAD_URL" "$FLASHHEAD_REF"; then
      fetched=1
      break
    fi
    sleep 5
  done
  if [ "$fetched" = 1 ]; then
    git -C "$UPSTREAM_DIR" -c advice.detachedHead=false checkout -q --force FETCH_HEAD || true
  fi
fi
if ! upstream_at_ref || [ ! -f "$UPSTREAM_DIR/flash_head/src/pipeline/flash_head_pipeline.py" ]; then
  echo "FATAL: SoulX-FlashHead is not at $FLASHHEAD_REF or has no pipeline code."
  echo "         Check the log above for git errors and restart."
  exit 1
fi
echo "SoulX-FlashHead: $(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)"

# FlashHead's Python packages go into ComfyUI's persistent virtualenv, the interpreter
# that runs ComfyUI. A bare `pip` would install into the system Python, where ComfyUI
# never sees them. Create the venv exactly as the image's /start.sh does when it is
# missing, so start.sh then just activates it. PIP_CONSTRAINT pins torch/torchvision/
# torchaudio to the image's builds: nothing here may replace them.
# Only what a single GPU needs: upstream's requirements.txt also pins gradio, xfuser,
# xformers and mediapipe for its demos and multi-GPU mode, which would disturb the
# image's environment. The pack stubs xfuser and mediapipe, and crops faces with
# OpenCV's YuNet (OpenCV >= 4.5.4, already in the image).
FLASHHEAD_PACKAGES=("diffusers==0.36.0" "loguru" "pyloudnorm" "einops" "safetensors")
VENV_DIR="$COMFYUI_PATH/${COMFYUI_VENV_NAME:-.venv-cu128}"
CONSTRAINTS=/opt/comfyui-runtime-constraints.txt
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating ComfyUI virtualenv $VENV_DIR (as the image's start.sh does)..."
  python3.12 -m venv --system-site-packages --without-pip "$VENV_DIR"
fi
VPY="$VENV_DIR/bin/python"
deps_ok() {
  "$VPY" - "$VENV_DIR" <<'PY'
import importlib, os, sys
venv = os.path.realpath(sys.argv[1])
for name in ("diffusers", "loguru", "pyloudnorm", "einops", "safetensors",
             "transformers", "torchaudio", "cv2", "PIL"):
    importlib.import_module(name)
import cv2, diffusers
if not hasattr(cv2, "FaceDetectorYN"):
    sys.exit(f"OpenCV {cv2.__version__} has no FaceDetectorYN (needs 4.5.4 or newer).")
if diffusers.__version__ != "0.36.0":
    sys.exit(f"diffusers {diffusers.__version__} is installed; the pack is pinned to 0.36.0.")
import torch
if os.path.realpath(torch.__file__).startswith(venv + os.sep):
    sys.exit("PyTorch was installed inside the venv; it must come from the image.")
PY
}
PKG_HASH="$(printf '%s\n' "${FLASHHEAD_PACKAGES[@]}" | sha256sum | cut -d' ' -f1)"
MARKER="$VENV_DIR/.flashhead-pro-avatar-deps"
if [ "$(cat "$MARKER" 2>/dev/null)" = "$PKG_HASH" ] && deps_ok 2>/dev/null; then
  echo "FlashHead Python packages already installed."
else
  echo "Installing FlashHead Python packages into $VENV_DIR..."
  constraint_args=()
  if [ -f "$CONSTRAINTS" ]; then
    constraint_args=(-c "$CONSTRAINTS")
  else
    # Older images: pin to whatever PyTorch the image ships.
    CONSTRAINTS="$(mktemp /tmp/flashhead-pro-torch-pins.XXXXXX)"
    "$VPY" -c 'import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    try: print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError: pass' > "$CONSTRAINTS"
    constraint_args=(-c "$CONSTRAINTS")
  fi
  if ! "$VPY" -m pip install --no-cache-dir "${constraint_args[@]}" "${FLASHHEAD_PACKAGES[@]}"; then
    echo "FATAL: installing FlashHead's Python packages failed (see pip output above)."
    echo "         Restart the pod to retry."
    exit 1
  fi
  # opencv-python needs libGL, which slim images can lack.
  if ! "$VPY" -c "import cv2" 2>/dev/null && command -v apt-get >/dev/null; then
    echo "cv2 does not import; installing the system libraries OpenCV needs..."
    apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 || true
  fi
  if ! deps_ok; then
    echo "FATAL: FlashHead's packages do not import in $VENV_DIR (see above)."
    exit 1
  fi
  printf '%s\n' "$PKG_HASH" > "$MARKER"
fi

# 2. Model folders, inputs and samples
echo "Preparing model directories..."
MODELS_DIR="$COMFYUI_PATH/models/flashhead"
mkdir -p "$MODELS_DIR/SoulX-FlashHead-1_3B" "$MODELS_DIR/wav2vec2-base-960h"
# The batch workflow renders every voiceover dropped in here, one video per file.
mkdir -p "$COMFYUI_PATH/input/batch_audio"
# Sample portrait and voiceover (71 s) the workflows open on. Never overwrites a file
# of the same name already on the volume.
for sample in "$SCRIPT_DIR/samples/"*; do
  [ -e "$COMFYUI_PATH/input/$(basename "$sample")" ] || cp "$sample" "$COMFYUI_PATH/input/"
done

# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------
# `hf` is only a downloader - ComfyUI never imports it - so it lives in its own venv.
# Upgrading huggingface_hub inside ComfyUI's environment breaks transformers' version
# pin, and ComfyUI then fails to start with "from transformers import CLIPTokenizer".
# Its Xet backend is much faster than wget on multi-GB files but cannot resume a
# partial file, so wget stays as the fallback.
echo "Installing fast downloader (optional)..."
HF_VENV="/workspace/.hf-cli"
HF_BIN=""
if [ ! -x "$HF_VENV/bin/hf" ]; then
  if python3 -m venv "$HF_VENV" 2>/dev/null; then
    "$HF_VENV/bin/pip" install -q -U "huggingface_hub[hf-xet]" 2>/dev/null || true
  fi
fi
for cand in "$HF_VENV/bin/hf" "$HF_VENV/bin/huggingface-cli"; do
  [ -x "$cand" ] && { HF_BIN="$cand"; break; }
done
if [ -n "$HF_BIN" ]; then
  echo "Fast downloader: $HF_BIN (isolated venv)"
else
  echo "Fast downloader unavailable, using wget."
fi

# A complete file: a safetensors header that accounts for every byte, a PyTorch
# checkpoint whose zip directory is intact (Wan2.1_VAE.pth), or valid JSON.
file_ok() {
  [ -f "$1" ] || return 1
  "$PY" - "$1" <<'PY'
import json, os, struct, sys
path = sys.argv[1]
kind = path[:-len(".part")] if path.endswith(".part") else path  # wget's staging name
try:
    if kind.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            json.load(f)
        sys.exit(0)
    if kind.endswith(".pth"):
        import zipfile
        # torch.save writes a zip whose central directory sits at the very end, so a
        # truncated download has none; data.pkl is the checkpoint's object graph.
        if os.path.getsize(path) < 1000000 or not zipfile.is_zipfile(path):
            raise ValueError('not a complete PyTorch checkpoint')
        with zipfile.ZipFile(path) as z:
            if not any(n.endswith('/data.pkl') for n in z.namelist()):
                raise ValueError('no data.pkl')
        sys.exit(0)
    if os.path.getsize(path) < 1000000:
        raise ValueError('too small')
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= size <= min(os.path.getsize(path) - 8, 100_000_000):
            raise ValueError('invalid header length')
        header = json.loads(f.read(size))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if end + size + 8 != os.path.getsize(path):
            raise ValueError('incomplete tensor data')
except (OSError, ValueError, KeyError, TypeError, struct.error):
    sys.exit(1)
PY
}

hf_fast() {
  # $1 = repo, $2 = path inside the repo, $3 = destination file
  local repo="$1" rpath="$2" dest="$3" tmp
  [ -n "$HF_BIN" ] || return 1
  # Stage beside the destination so the final move is a rename, not a multi-GB copy.
  tmp="$(mktemp -d "$(dirname "$dest")/.hfdl.XXXXXX")" || return 1
  if HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$repo" "$rpath" --local-dir "$tmp"; then
    file_ok "$tmp/$rpath" && mv -f "$tmp/$rpath" "$dest"
  fi
  rm -rf "$tmp"
  file_ok "$dest"
}

fetch_url() {
  # $1 = destination file, $2 = URL
  local dest="$1" url="$2" part="$1.part"
  if [ -n "$HF_TOKEN" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$part" "$url" || return 1
  else
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$part" "$url" || return 1
  fi
  file_ok "$part" && mv -f "$part" "$dest"
}

download_hf() {
  # $1 = Hugging Face repo, $2 = path inside it, $3 = folder under models/flashhead.
  # Files keep their path inside the repo (Model_Pro/..., VAE_Wan/...), the layout
  # upstream's pipeline loads.
  local repo="$1" rpath="$2" dest="$MODELS_DIR/$3/$2"
  mkdir -p "$(dirname "$dest")"
  if file_ok "$dest"; then
    echo "$3/$rpath already exists, skipping."
    return 0
  fi
  echo "Downloading $3/$rpath ..."
  hf_fast "$repo" "$rpath" "$dest" && { echo "$3/$rpath done."; return 0; }
  echo "  falling back to wget..."
  fetch_url "$dest" "https://huggingface.co/$repo/resolve/main/$rpath" || true
  if ! file_ok "$dest"; then
    echo "ERROR: could not download $3/$rpath. Existing files were retained for retry."
    return 1
  fi
  echo "$3/$rpath done."
}

# -------------------------------------------------------------------
# 3. Model downloads (~6.9 GB)
# -------------------------------------------------------------------
FAILED=0
FH_REPO="Soul-AILab/SoulX-FlashHead-1_3B"
W2V_REPO="facebook/wav2vec2-base-960h"

# FlashHead Pro (1.3B DiT, ~6.0 GB) and the Wan 2.1 VAE it decodes with (~0.5 GB).
# All Apache-2.0.
download_hf "$FH_REPO" "Model_Pro/config.json" SoulX-FlashHead-1_3B || FAILED=1
download_hf "$FH_REPO" "Model_Pro/diffusion_pytorch_model.safetensors" SoulX-FlashHead-1_3B || FAILED=1
download_hf "$FH_REPO" "VAE_Wan/Wan2.1_VAE.pth" SoulX-FlashHead-1_3B || FAILED=1
# wav2vec2-base-960h audio encoder (~0.4 GB): weights, config and feature extractor.
download_hf "$W2V_REPO" "config.json" wav2vec2-base-960h || FAILED=1
download_hf "$W2V_REPO" "preprocessor_config.json" wav2vec2-base-960h || FAILED=1
download_hf "$W2V_REPO" "model.safetensors" wav2vec2-base-960h || FAILED=1

if [ "$FAILED" = "1" ]; then
  echo "FATAL: one or more model downloads failed (see ERROR lines above)."
  echo "         Restart the pod to retry - finished files are kept and skipped."
  exit 1
fi

# 4. Install bundled workflows into ComfyUI's Workflows sidebar
echo "Installing workflows..."
WF_DEST="$COMFYUI_PATH/user/default/workflows"
mkdir -p "$WF_DEST"
# Workflows shipped under earlier names, and earlier versions of current ones. Each is
# removed only while it is byte-identical to that shipped version, so a copy edited on
# the pod is kept; a current name is then reinstalled at its new version below.
RETIRED_WORKFLOWS="
c28a3b51e50dd215038f70f441cd1f4008f5164b2008d915d15dbc3fced61ed6 flashhead_pro_1_continuous.json
8da695c13dca5143f4067336d00f4c20d6db0785e1a11ee07e31cc5f503f617f flashhead_pro_2_refresh_at_pauses.json
4fb8eb175aa3182d61fc543ea9d9f76ed53d9189c294e2aa374dda60af14058e flashhead_pro_3_batch_folder_refresh_at_pauses.json
45b6f3ebc237f1e6f4391a15cf6d8d6e396c7a13a48c8f51dc086b69d0e276d5 flashhead_pro_1_continuous.json
f348abd689cd91a11d51a2880eede29d3c6f83a16e6b336d3996105b7cd77e35 flashhead_pro_2_refresh_at_pauses.json
66ae8e878759cf77d2307f3a8aeff126eb6741bc3d55ca6ed49c597f11584aa9 flashhead_pro_3_batch_folder_refresh_at_pauses.json
7d6beb9ee006893969261146b1db57180955d246cdcedd883d393ce79fe1b51e flashhead_pro_1_continuous.json
c81d4c93b4aa76ec0b81980642bda0a9eeab58376e4f071938bd19a6417c7fa7 flashhead_pro_2_refresh_at_pauses.json
4f13172db4a8ade5b55bd6625b74f2fdcab9167e6849382191fedd1acb097199 flashhead_pro_3_batch_folder_refresh_at_pauses.json
15b44e38f49269b564a5ec323979c6b30fb9018c1ad1521e8df309891dce525b flashhead_pro_1_continuous.json
ab811dcb25afd6541e0d771e0f9592496f38df21d9ab2f8105629df6c5652364 flashhead_pro_2_refresh_at_pauses.json
d8e8261cbd36cf363bd9dcfdf61000b227697b6ca67b9aa3ea1e57d124bf90f8 flashhead_pro_3_batch_folder_refresh_at_pauses.json
"
while read -r sum name; do
  [ -n "$name" ] && [ -f "$WF_DEST/$name" ] || continue
  if [ "$(sha256sum "$WF_DEST/$name" | cut -d' ' -f1)" = "$sum" ]; then
    rm -f "$WF_DEST/$name"
    echo "  removed outdated workflow: $name"
  fi
done <<< "$RETIRED_WORKFLOWS"
if [ -d "$SCRIPT_DIR/workflows" ]; then
  # Never overwrite a workflow already edited on the Network Volume.
  for workflow in "$SCRIPT_DIR/workflows/"*.json; do
    [ -e "$WF_DEST/$(basename "$workflow")" ] || cp "$workflow" "$WF_DEST/"
  done
  ls "$WF_DEST" | sed 's/^/  workflow: /'
else
  echo "No workflows directory found in repo, skipping."
fi

# 5. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
exec /start.sh
