# FlashHead Pro Avatar - Long Talking Heads, One GPU

Turn a portrait and voiceover of any length into a 512 x 512 talking-head video with SoulX-FlashHead Pro (1.3B), stitched with your original audio. Pro is FlashHead's higher-quality model; one run renders the whole voiceover.

## Storage

Attach your own **Network Volume at /workspace** when deploying (**50 GB recommended**). The container disk is 5 GB and the Volume disk 0 GB.

ComfyUI, models, uploads, workflows and renders stay on the volume. Reattach it in its datacenter when replacing a pod.

## Connect

- **Port 8188:** ComfyUI
- **Port 8888:** JupyterLab

First boot downloads **6.9 GB**. Wait for "Setup complete!" in logs.

Jupyter has no login by default; anyone with its URL can run commands. For a login, set JUPYTER_NO_AUTH=0 and JUPYTER_PASSWORD.

## Generate a video

1. Open ComfyUI's Workflows sidebar.
2. Select **flashhead_pro_1_continuous**.
3. Load your portrait and voiceover (a 71 s sample is preloaded).
4. Check the portrait preview: **face crop** (as FlashHead was trained) or **whole image**.
5. Run. The first run loads the model and warms up for a minute or two.

The video lands in ComfyUI/output; timing goes to output/flashhead_stats/.

## Long voiceovers

FlashHead renders 1.12 s chunks, each continuing from the previous one. For a minute or more, use workflow **2**: every 20-40 s, in the middle of a real pause (mouth shut), it restarts from the portrait so the face cannot drift. The cuts are planned from the voiceover up front. **3** does the same for every voiceover in input/batch_audio/.

## Hardware

Pro renders about 2.3 s per second of video on an RTX 4090 (upstream). Use an RTX 4090 or 5090 (12 GB+ VRAM).

All models are Apache-2.0.

Full instructions:
https://github.com/tenitsky/flashhead-pro-avatar
