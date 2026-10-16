# FlashHead Pro Avatar - Long Talking Heads, One GPU

Turn **one portrait and a voiceover of any length** into a 512 x 512, 25 fps talking-head video with **SoulX-FlashHead Pro** (1.3B), stitched with your original audio. Sharp eyes, teeth and hair.

## Storage

Attach your own volume at /workspace:

- **Network Volume** (50 GB): everything stays on it. Reattach it in its datacenter.
- **Global Volume** (any datacenter): models, uploads and renders stay on it; ComfyUI is set up again on each new pod (a few minutes). Container disk **20 GB**.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab (uploads, downloads, terminal)

First boot downloads **6.9 GB**; wait for **"Setup complete!"** in the logs. Jupyter has no login by default; set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD for one.

## Generate a video

1. Open the **Workflows** sidebar and select **flashhead_pro_1_continuous**.
2. Set **Load Portrait** and **Load Voiceover** (samples are preloaded).
3. **Run.** The face is framed automatically.

The video lands in ComfyUI/output/. It is saved in ~30 s parts as it renders; cancel and queue again to resume.

**flashhead_pro_2_batch_folder_continuous** renders every voiceover in ComfyUI/input/batch_audio/.

## Hardware

About **2.3 s per second of video on an RTX 4090**. Use a 4090, 5090, L40S, A100 or H100 (12 GB+ VRAM). All models are Apache-2.0; no token needed.

Full instructions:
https://github.com/tenitsky/flashhead-pro-avatar
