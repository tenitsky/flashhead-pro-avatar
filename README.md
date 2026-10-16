# SoulX-FlashHead Pro: RunPod ComfyUI template (long talking head, one GPU)

A one-shot template that installs **SoulX-FlashHead Pro** (Soul-AILab, 1.3B) on RunPod
with ComfyUI and JupyterLab. It turns **one portrait and a voiceover of any length** into
a 512 × 512 talking-head video at 25 fps, stitched with your original audio, on a
**single GPU**.

**Pro vs. Lite.** Both are 1.3B models at 512 × 512. Lite decodes with the LTX-Video VAE,
which compresses 32 × 32 pixels and 8 frames into one latent: very fast, but soft. Pro
decodes with the Wan 2.1 VAE (8 × 8 pixels, 4 frames), so it renders far more detail
at roughly 4–6× Lite's cost. Upstream: Pro runs at 10.8 FPS on an RTX 4090 (about
2.3 s per second of video); Lite at 96 FPS.

Both workflows render **continuously**: every chunk flows on from the one before, as
upstream does, with no reset to the portrait. One renders a single voiceover, the other
a folder of them.

ComfyUI has no FlashHead nodes. This repo's small `comfyui-flashhead` pack drives
upstream's own pipeline code, which setup installs at a pinned commit.

## How FlashHead joins chunks

FlashHead generates **33-frame chunks**. The first latents of each chunk are the
previous chunk's last **5 frames** (2 latents of Pro's Wan VAE), re-encoded: video
continuation, as in LongCat, not H3's independent chunks. Each chunk therefore adds
**28 new frames (1.12 s)**, and the
portrait is the identity reference for every chunk. The audio goes through wav2vec2 in a
rolling 8-second window ending where the chunk ends, exactly as upstream streams it.

The **anchor** setting on the Render node controls what a chunk opens on:

| Anchor | Each chunk opens on | Trade-off |
|---|---|---|
| **continuous (upstream)** | the previous chunk's last 5 frames | seamless joins; could drift over long videos |
| refresh from portrait at a pause | continuous, but every 40-60 s the portrait, in the middle of a real pause | one small jump in a silence per 40-60 s; drift bounded |
| reset to portrait every chunk / every N seconds | the portrait on a fixed schedule | for comparison: jumps regardless of the speech |

The shipped workflows use **continuous**; the other anchors stay available on the
Render node. A reset is upstream's own `reset_person_name()`: the next chunk starts the
way the first one does.

For **refresh at a pause**, the refreshes are **planned from the whole voiceover before
rendering starts**, and only go where the mouth is shut:

1. Every **real pause** is found first: at least **0.4 s** of near-silence, measured
   against the recording's own speech and noise levels. Short gaps between words do
   not count.
2. Each refresh goes **in the middle of the longest real pause** between `reset_seconds`
   (40) and `reset_seconds` + 20 s after the previous one, never in the last 3 s.
3. If that window has no real pause, the **next real pause within 20 s** is used rather
   than cutting into speech. Only when there is none at all does a shorter gap get it.

The chunk before a refresh keeps only its frames up to that point, so the refresh lands
on the exact frame, not on the nearest 1.12 s chunk boundary. The log lists each refresh
with its pause length. On the 57 s sample, the one refresh lands at 52.2 s, in the middle of a
0.92 s pause.

### What a first test showed (an earlier 71 s voiceover, same seed as Lite)

Rendered with the sample portrait (face crop) and the template's earlier 71 s sample
voiceover, continuous,
and measured with [`tools/analyze_seams.py`](tools/analyze_seams.py):

| | Pro | Lite (for comparison) |
|---|---|---|
| Detail | sharp eyes, lashes, freckles, separate teeth, hair strands | soft: skin and hair smeared, teeth blur together |
| Change across a join vs. a normal frame | **1.23×** (seamless) | 1.2× |
| Distance from the portrait per 10 s | 2.5–11, no upward trend | 4–8, no upward trend |
| Colour over 71 s | constant | constant |

Pro is the clear quality step up, and it keeps Lite's seamless joins with no
measurable drift over 71 s. Minutes-long runs are the next test.

## What boots

On first pod boot, `setup.sh`:

1. Works out which volume is attached: a **Network Volume** or a **Global Volume**
   (see [Storage](#storage-network-volume-or-global-volume)); refuses to install on the
   container disk alone
2. Copies the baked ComfyUI from `/opt/comfyui-baked` to `/workspace/runpod-slim/ComfyUI`
   (self-healing, skip-if-present). With a Global Volume it goes to the container disk
   instead, and its `models`, `input`, `output` and `user/default` folders link to the
   volume
3. Checks that the image's PyTorch has a CUDA build that can use the GPU; stops if not,
   before any download
4. Installs the `comfyui-flashhead` nodes and **SoulX-FlashHead at a pinned commit**
   inside them, then adds the few Python packages a single GPU needs to ComfyUI's
   persistent virtualenv, with the image's PyTorch pinned
   (`/opt/comfyui-runtime-constraints.txt`)
5. Copies the sample portrait and voiceover into `ComfyUI/input/` (never over a file of
   the same name)
6. Downloads the model files (skip-if-exists, resumable, validated)
7. Installs the two workflows into ComfyUI's **Workflows** sidebar
8. Hands over to `/start.sh` (ComfyUI on port **8188**, JupyterLab on **8888**)

On later boots every step finds its work done and is skipped. With a Global Volume, a
fresh container disk repeats steps 2 and 4 (a few minutes: copy ComfyUI, fetch upstream's
code, install the packages); the models are never downloaded again.

Upstream's `requirements.txt` is **not** installed: it pins gradio, xformers, xfuser and
mediapipe for its demos and multi-GPU mode, which would disturb the image. On one GPU
xfuser and mediapipe are never called, so the pack supplies small stand-ins, and faces
are found with OpenCV's YuNet detector (shipped, MIT licence) instead of mediapipe.

## Models pulled

| File (under `ComfyUI/models/flashhead/`) | Size | From |
|---|---|---|
| `SoulX-FlashHead-1_3B/Model_Pro/` (config + weights) | 6.0 GB | [Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth` | 0.5 GB | same repo |
| `wav2vec2-base-960h/` (config, feature extractor, weights) | 0.4 GB | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) |

About **6.9 GB**, all ungated and **all Apache-2.0** (the FlashHead DiT, the Wan 2.1
VAE and wav2vec2; Pro avoids Lite's Open-RAIL-M LTX VAE). Use **50 GB of Network Volume storage** as a starting
point for ComfyUI, its environment, models and renders. A Global Volume grows by itself.

## Storage: Network Volume or Global Volume

Attach **one** of these at `/workspace` when deploying. Setup detects which it is and
says so in the log (`Storage: network volume ...` or `Storage: global volume ...`).

| | Network Volume | Global Volume (RunPod beta) |
|---|---|---|
| Where pods can run | its datacenter only | any datacenter (GPU pods) |
| ComfyUI, venv, node pack, upstream code | on the volume | on the container disk, set up again on each fresh container (a few minutes) |
| Models, inputs, outputs and parts, saved workflows | on the volume | on the volume |
| Size | fixed, 50 GB recommended | grows by itself |
| Container disk | 5 GB is enough | **20 GB** (first-boot downloads are staged there) |

A [Global Volume](https://docs.runpod.io/storage/globalvolume/overview) is not tied to a
datacenter, but it is object storage: **no file locking, atomic rename, hard links or
permission bits**. A Python virtualenv, a git checkout and pip depend on those, so setup
splits the work:

| Data | Location with a Global Volume |
|---|---|
| ComfyUI, `comfyui-flashhead`, SoulX-FlashHead code, Python environment | container disk: `/opt/flashhead-local/ComfyUI/` |
| Models | volume: `runpod-slim/ComfyUI/models/flashhead/` |
| Portraits, voiceovers, `batch_audio/` | volume: `runpod-slim/ComfyUI/input/` |
| Finished videos and the parts of an unfinished one | volume: `runpod-slim/ComfyUI/output/` |
| Saved workflows and ComfyUI settings | volume: `runpod-slim/ComfyUI/user/default/` |
| Render scratch files | container disk (ComfyUI's temp folder) |
| Jupyter settings, IPython history, download caches, FileBrowser database | container disk: `/opt/flashhead-local/` |

The volume paths are the same as on a Network Volume, so a Network Volume copied over
with RunPod's suggested `rsync -a /workspace/ /workspace-global/` is picked up as it is.
Downloads are staged on the container disk, checked, then copied to their final name;
nothing is renamed or locked on the volume. The image's `/start.sh` fixes ComfyUI at
`/workspace/runpod-slim/ComfyUI`, so setup runs a copy of it pointed at the
container-disk install (`/opt/flashhead-local/image-start.sh`); the image file itself is
not edited.

Things to know:

- **A fresh container disk takes a few minutes longer to boot**: ComfyUI, the node pack
  and its packages are set up again. Models are not downloaded again.
- **Custom nodes you add with ComfyUI-Manager do not persist** (container disk).
- **Run one pod at a time on a fresh volume** until `Setup complete!`; two pods writing
  the same file on a Global Volume overwrite each other. Give each pod its own voiceovers.
- Mounted at `/workspace-global` instead (next to a Network Volume, or with its mount path
  edited), setup finds it there. A Network Volume at `/workspace` takes priority;
  `FLASHHEAD_STORAGE=global` uses the Global Volume instead, `FLASHHEAD_STORAGE=network`
  forces the Network Volume layout.

Global Volumes are in beta at RunPod; their behaviour may change.

## RunPod public template settings

| Setting | Value |
|---|---|
| Template name | `FlashHead Pro Avatar - ComfyUI + JupyterLab` |
| Compute type | **NVIDIA / GPU** |
| Base image | **`runpod/comfyui:1.4.0-comfyuiv0.35.0-cuda12.8`** |
| Registry authentication | None; image is public |
| Container disk | **20 GB** (a Global Volume stages downloads there; a Network Volume needs only 5 GB) |
| Persistent storage in the public template editor | **Volume disk**, **0 GB** |
| Template volume mount path | **`/workspace`** |
| Volume at deployment | Each deployer attaches their own **Network Volume** (50 GB recommended) or **Global Volume** at `/workspace` |
| HTTP ports | **`8188,8888`** (ComfyUI, JupyterLab) |
| TCP ports | None required; expose `22` only if SSH is wanted |
| Environment variables | `JUPYTER_NO_AUTH=1`; `HF_TOKEN` optional |
| README tab | Paste [`TEMPLATE_README.md`](TEMPLATE_README.md) |

The image ships PyTorch 2.10 (CUDA 12.8) on Python 3.12, which runs on RTX 4090/5090,
L40S, A100 and H100 hosts (driver r570 and newer). If a host's driver is too old,
setup stops before downloading anything and says so.

`JUPYTER_NO_AUTH=1` runs Jupyter without a login: setup passes the image's own
`JUPYTER_DISABLE_AUTH=true` switch to `/start.sh`. For authenticated Jupyter, use
`JUPYTER_NO_AUTH=0` and set `JUPYTER_PASSWORD` at deployment; it becomes the token.
Do not override `COMFYUI_PATH`: the image starts ComfyUI at its fixed path.

**Container start command.** Paste this JSON into RunPod's container start command
field (also provided as [`runpod-start.json`](runpod-start.json); readable source:
[`bootstrap.sh`](bootstrap.sh)). It checks DNS for GitHub and Hugging Face first and
leaves working DNS unchanged. If two checks fail, it saves the original
`/etc/resolv.conf` under `/tmp/flashhead-pro-avatar-dns.*`, then tries the public resolvers
`1.1.1.1` and `8.8.8.8`; if resolution still fails, it restores the original and stops
with a networking error. Git clone failures are retried up to 10 times. It clones
`main` unless `TEMPLATE_REF` names another branch or tag: `TEMPLATE_REF=v1-network-volume`
runs the version from before Global Volume support.

```json
{
  "entrypoint": [
    "bash",
    "-lc"
  ],
  "cmd": [
    "set -e\ndns_ok() {\n  timeout 10 getent ahostsv4 github.com >/dev/null 2>&1 &&\n  timeout 10 getent ahostsv4 huggingface.co >/dev/null 2>&1\n}\nif ! dns_ok; then\n  echo 'DNS lookup failed; retrying in 3 seconds...'\n  sleep 3\n  if ! dns_ok; then\n    dns_backup=$(mktemp /tmp/flashhead-pro-avatar-dns.XXXXXX)\n    cp /etc/resolv.conf \"$dns_backup\"\n    echo \"Trying public DNS; original resolver saved at $dns_backup\"\n    if { printf 'nameserver 1.1.1.1\\nnameserver 8.8.8.8\\noptions timeout:2 attempts:2\\n'; } > /etc/resolv.conf && dns_ok; then\n      echo 'DNS recovered; continuing setup.'\n    else\n      cat \"$dns_backup\" > /etc/resolv.conf || true\n      echo 'FATAL: DNS still unavailable or resolver is read-only. Check RunPod networking or deploy on another host with the same Network Volume.' >&2\n      exit 1\n    fi\n  fi\nfi\nfor attempt in 1 2 3 4 5 6 7 8 9 10; do\n  repo=$(mktemp -d /tmp/flashhead-pro-avatar.XXXXXX)\n  if git clone --depth 1 --branch \"${TEMPLATE_REF:-main}\" https://github.com/tenitsky/flashhead-pro-avatar.git \"$repo\"; then\n    exec bash \"$repo/setup.sh\"\n  fi\n  sleep 5\ndone\necho 'FATAL: unable to clone flashhead-pro-avatar after 10 attempts' >&2\nexit 1\n"
  ]
}
```

For the first launch, check the logs for **`Network Volume workspace ready at
/workspace`** (or **`Global Volume ready at /workspace`**), then `SoulX-FlashHead: 9bc03de`, the model downloads and **`Setup
complete!`**.

## After it boots

1. Open ComfyUI (port 8188), open the **Workflows** sidebar and choose
   **`flashhead_pro_1_continuous`**.
2. **Load Portrait** and **Load Voiceover** open on the bundled samples
   (`sample_portrait.png`, `sample_voiceover.mp3`, 57 s), so the first run needs no
   uploads.
3. **Run.** One run renders the whole voiceover; the first run also loads the model and,
   with `compile` on, spends a minute or two warming up. The Render node shows the
   newest frame as it renders, and finished parts are playable straight away (below).
4. The video is saved as `output/<audio name><suffix>.mp4` and shown on the Render node.
   That is all that stays in `output`. Timing (seconds of rendering per second of video,
   peak VRAM) is printed in the log; scratch files go to ComfyUI's temp folder, which
   ComfyUI empties at every start.

To measure seams and drift of a render from a Jupyter terminal:

```bash
cd /workspace/runpod-slim/ComfyUI
python /tmp/flashhead-pro-avatar.*/tools/analyze_seams.py output/sample_voiceover_pro_continuous.mp4
```

> `setup.sh` skips existing workflows, so a workflow you edited on the pod is never
> overwritten on reboot. A file still identical to a version this repo shipped earlier
> is replaced with the current version; edited copies are kept.

### Watching a long render, and resuming it

A 20-minute voiceover is one run, but you don't have to wait for the end to see it:

- **Live frame.** The Render node shows the newest frame after every 1.12 s chunk.
- **Parts.** The video is saved in **~30 s parts** as it renders, each a playable mp4
  with its slice of the voiceover:
  `output/flashhead_parts/<audio name><suffix>/part_001.mp4`, `part_002.mp4`, … Open
  them in JupyterLab (port 8888) while the rest renders. A part ends at the first chunk
  boundary after 30 s, or where the render restarts from the portrait.
- **Resume.** Next to each part, the state the next chunk continues from is saved
  (upstream's carried motion latents and random generator). If you cancel the run, or
  the pod stops, queue it again with the same inputs: it picks up after the last
  finished part and the result is identical to an uninterrupted render. Another
  voiceover, portrait, seed or anchor starts fresh.
- When every part is done they are joined without re-encoding, the original audio is
  laid over the whole video, and the parts are removed. `output/flashhead_parts/` only
  exists while a render is unfinished.

ComfyUI has no pause button: **cancel** is the pause, and queueing again resumes.

### Workflows

| Workflow | Anchor | Output suffix |
|---|---|---|
| `flashhead_pro_1_continuous` | continuous (upstream) | `_pro_continuous` |
| `flashhead_pro_2_batch_folder_continuous` | continuous, for every voiceover in `input/batch_audio/` | `_pro_continuous` |
Neither resets to the portrait. The batch version renders every file in the folder in
one run, one video each, skipping files already rendered.

**Framing.** The Render node finds the face (OpenCV's YuNet) and frames the portrait as
FlashHead was trained: a 512 × 512 square twice the face width, the face centred, 55% of
the square above its centre. Without a face, the image is center-cropped. Any portrait
size works.

The anchor, reset spacing, seed and colour correction are all on the Render node, so
any workflow can be switched; change `name_suffix` with them so outputs do not overwrite
each other. **color_correction** (upstream 1.0) pulls every chunk's colours toward the
portrait.

### Without ComfyUI

[`tools/render_local.py`](tools/render_local.py) runs the same engine from a terminal,
on the pod or on any CUDA machine with the models:

```bash
python tools/render_local.py --portrait my.png --audio my.mp3 --out my.mp4 \
    --anchor continuous --framing face
```

## Hardware

Upstream: Pro renders **10.8 FPS on one RTX 4090**, about **2.3 s per second of
video**, and real time on two RTX 5090s with SageAttention. An RTX 4090 or 5090 pod is
the natural choice; an H100 should be faster still. Peak VRAM in the local test was
**6.2 GB** allocated, but give it **12 GB or more**: on an 8 GB laptop GPU, Windows
moved part of the model into system memory, and rendering took 12-79 s per second of
video depending on memory pressure, which says nothing about a pod.

## Template env vars (all optional)

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Not required (both repos are ungated). It raises Hugging Face rate limits |
| `JUPYTER_NO_AUTH` | `1` | Disables Jupyter login (anyone with the URL gets a shell). Set `0` and `JUPYTER_PASSWORD` for a login |
| `JUPYTER_PASSWORD` | *(empty)* | Jupyter token, used when `JUPYTER_NO_AUTH=0` |
| `FLASHHEAD_REF` | pinned commit | Test a newer SoulX-FlashHead commit. The engine was built against the pin |
| `FILEBROWSER_PASSWORD` | `adminadmin12` | FileBrowser on port 8080 (user `admin`), if you expose it. **Change the default** |
| `COMFYUI_PATH` | `/workspace/runpod-slim/ComfyUI` | Leave unset; must match the base image's fixed startup path |
| `HF_HOME` | `/workspace/.cache/huggingface` | Keeps the HF cache on the volume, not the container disk (Global Volume: `/opt/flashhead-local/.cache/huggingface`) |
| `FLASHHEAD_STORAGE` | `auto` | `network` or `global` overrides which volume setup thinks it has |
| `TEMPLATE_REF` | `main` | Branch or tag of this repo the start command clones; `v1-network-volume` is the version before Global Volume support |

## What has and hasn't been verified

Verified: FlashHead Pro rendering a full 71 s voiceover (continuous, face crop) on a
local RTX 3080 Laptop GPU (PyTorch 2.10 cu128, diffusers 0.36.0, OpenCV 5.0), through
the same engine the nodes use, with the seam and drift measurements above; the shipped
Pro workflow run in ComfyUI v0.35.0; the Wan 2.1 VAE checkpoint's format check. On CPU in CI: the
chunk plan, audio windows and reset schedule against a recording fake pipeline; the
parts, the live preview and a cancelled render resuming to an identical video; joining
parts frame-exact with ffmpeg; the face
crop on the sample; the stand-ins; the ffmpeg output; that the pinned upstream commit
still has the API, settings and reset the engine relies on; the workflows against the
nodes' schemas; and a full first boot and reboot of `setup.sh` against an offline image
fixture, on a Network Volume and on a simulated Global Volume that refuses rename, hard
links, symlinks and permission changes.

**Not yet verified:** a boot on a real RunPod pod (Network or Global Volume), `torch.compile` on Linux, and runs of
several minutes.

## Repository layout

```
├── setup.sh                               # RunPod boot script
├── bootstrap.sh / runpod-start.json       # container start command (readable / paste-ready)
├── README.md
├── TEMPLATE_README.md                     # paste into the public template's README tab
├── RUNPOD_TEMPLATE_SETUP.txt              # every template field, with comments and links
├── custom_nodes/
│   └── comfyui-flashhead/                 # Load Model / Render / Batch Folder
│       ├── engine.py                      # chunk loop, anchors, face crop, ffmpeg output
│       ├── storage.py                     # volume detection and checks (Network or Global Volume)
│       └── face_detection_yunet_2023mar.onnx
├── samples/                               # sample portrait + 57 s voiceover the workflows open on
├── tools/
│   ├── build_workflow.py                  # generates the workflow JSONs; re-run after edits
│   ├── render_local.py                    # the engine without ComfyUI
│   ├── analyze_seams.py                   # seam and drift measurements
│   ├── test_engine.py                     # chunk plan, audio, anchors, crop, output (CPU)
│   ├── test_upstream.py                   # the pinned upstream still matches the engine
│   ├── test_workflows.py                  # graphs vs. node schemas, samples, downloads
│   ├── test_storage.py                    # volume detection, Network and Global Volume checks
│   ├── test_boot.py                       # full setup with image/network fixtures (Linux)
│   ├── test_jupyter.py                    # auth switch and real Jupyter Server smoke test
│   └── test_setup.py                      # Bash download and start-command checks
└── workflows/                             # the two workflows listed above
```

## Credits

[SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) (Soul-AILab,
[technical report](https://arxiv.org/pdf/2602.07449)) ·
[YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
(OpenCV Zoo, MIT) · [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
(Meta, Apache-2.0)
