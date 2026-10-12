# FlashHead Pro Avatar - Long Talking Heads, One GPU

Turn **one portrait and a voiceover of any length** into a 512 x 512, 25 fps talking-head video with **SoulX-FlashHead Pro** (1.3B), stitched with your original audio. One run renders the whole voiceover. Pro is FlashHead's high-quality model: sharp eyes, teeth and hair.

## Storage

Attach your own **Network Volume at /workspace** when deploying (**50 GB recommended**). The container disk is 5 GB and the Volume disk 0 GB.

ComfyUI, models, uploads, workflows and renders stay on the volume. Reattach it in its datacenter when replacing a pod; later boots skip the downloads.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab (uploads, downloads, terminal)

First boot downloads **6.9 GB**. Wait for **"Setup complete!"** in the logs.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's **Workflows** sidebar.
2. Select **flashhead_pro_1_continuous**.
3. Set **Load Portrait** and **Load Voiceover**. A sample portrait and a 71 s sample voiceover are preloaded, so the first run needs no uploads.
4. Check the portrait preview. **face crop** frames a square twice the face width, as FlashHead was trained; **whole image** keeps shoulders and background (smaller face). face_ratio widens the crop.
5. **Run.** The first run loads the model and warms up for a minute or two.

The video is saved as ComfyUI/output/<audio name>_pro_continuous.mp4. Timing and peak VRAM go to output/flashhead_stats/.

## Watch it as it renders

- The Render node shows the **newest frame** live.
- The video is saved in **~30 s parts** you can play in JupyterLab while the rest renders: output/flashhead_parts/<name>/part_001.mp4, ...
- **Cancel** to stop; queue again with the same inputs and it **resumes** after the last finished part.

## Workflows

1. **Continuous** - every 1.12 s chunk continues from the previous one, as upstream: seamless joins. Best for clips up to a minute.
2. **Refresh at Pauses (20-40 s)** - continuous, plus a restart from the portrait once in every 20-40 s so the face cannot drift over minutes.
3. **Batch Folder** - workflow 2 for every voiceover in ComfyUI/input/batch_audio/ (upload with JupyterLab), one video each, with the same portrait.

## Restarts only in silence

The restarts are **planned from the whole voiceover before rendering**:

- A **real pause** is at least 0.4 s of near-silence, measured against the recording's own levels. Gaps between words do not count.
- Each restart goes **in the middle of the longest real pause** 20-40 s after the previous one; if there is none, the next real pause within 20 s.
- The chunk before it ends exactly there, so the small jump happens while the mouth is shut.

The log lists every restart with its pause length. reset_seconds on the Render node sets the earliest restart (20 s).

## Tips

- Use a clear, front-facing portrait with an open view of the mouth.
- Clean voiceovers (no music) give the best lip sync and pause detection.
- A workflow you edit is never overwritten on reboot.

## Hardware

Pro renders about **2.3 s per second of video on an RTX 4090** (upstream). Use an RTX 4090 or 5090, or an L40S, A100 or H100 (12 GB+ VRAM, driver r570+). If the driver is too old, setup stops before downloading.

All models are **Apache-2.0** and ungated; no Hugging Face token needed.

Full instructions:
https://github.com/tenitsky/flashhead-pro-avatar
