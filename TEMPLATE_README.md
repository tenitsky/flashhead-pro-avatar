# FlashHead Pro Avatar - Long Talking Heads, One GPU

Turn **one portrait and a voiceover of any length** into a 512 x 512, 25 fps talking-head video with **SoulX-FlashHead Pro** (1.3B), stitched with your original audio. One run renders the whole voiceover. Pro is FlashHead's high-quality model: sharp eyes, teeth and hair.

## Storage

Attach your own volume at /workspace when deploying. Either kind works; setup detects which:

- **Network Volume** (**50 GB recommended**): ComfyUI, models, uploads, workflows and renders stay on the volume. Reattach it in its datacenter when replacing a pod; later boots skip the downloads.
- **Global Volume** (beta, any datacenter): models, uploads, workflows and renders stay on the volume; ComfyUI itself runs from the container disk and is set up again on each fresh container (a few minutes). Keep the container disk at **20 GB**: first-boot downloads are staged there. Let the first pod finish downloading before starting others.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab (uploads, downloads, terminal)

First boot downloads **6.9 GB**. Wait for **"Setup complete!"** in the logs.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's **Workflows** sidebar.
2. Select **flashhead_pro_1_continuous**.
3. Set **Load Portrait** and **Load Voiceover**. A sample portrait and a 71 s sample voiceover are preloaded, so the first run needs no uploads.
4. **Run.** The first run loads the model and warms up for a minute or two.

The face is found and framed automatically, as FlashHead was trained: a 512 x 512 square twice the face width around it.

The video is saved as ComfyUI/output/<audio name>_pro_continuous.mp4. Timing and peak VRAM are printed in the log.

## Watch it as it renders

- The Render node shows the **newest frame** live.
- The video is saved in **~30 s parts** you can play in JupyterLab while the rest renders: output/flashhead_parts/<name>/part_001.mp4, ...
- **Cancel** to stop; queue again with the same inputs and it **resumes** after the last finished part.
- When the video is done, the parts are removed: output holds only finished videos.

## Workflows

1. **Continuous** - one voiceover. Every 1.12 s chunk continues from the previous one: seamless, never reset to the portrait.
2. **Batch Folder Continuous** - the same for every voiceover in ComfyUI/input/batch_audio/ (upload with JupyterLab), one video each. Files already rendered are skipped.

## Tips

- Use a clear, front-facing portrait with an open view of the mouth.
- Clean voiceovers (no music) give the best lip sync.
- A workflow you edit is never overwritten on reboot.

## Hardware

Pro renders about **2.3 s per second of video on an RTX 4090** (upstream). Use an RTX 4090 or 5090, or an L40S, A100 or H100 (12 GB+ VRAM, driver r570+). If the driver is too old, setup stops before downloading.

All models are **Apache-2.0** and ungated; no Hugging Face token needed.

Full instructions:
https://github.com/tenitsky/flashhead-pro-avatar
