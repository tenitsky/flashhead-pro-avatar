#!/usr/bin/env python3
"""
Render with the template's FlashHead engine without ComfyUI, e.g. from a Jupyter
terminal on the pod or on a local GPU:

    python tools/render_local.py --portrait samples/sample_portrait.png \
        --audio samples/sample_voiceover.mp3 --anchor continuous --out out.mp4

--models defaults to the pod's ComfyUI/models/flashhead. --upstream points at a
SoulX-FlashHead checkout (on the pod: the one inside the custom node). Writes a
<out>.json with the timing next to the video.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.join(HERE, "..", "custom_nodes", "comfyui-flashhead")
sys.path.insert(0, NODE)
import engine  # noqa: E402

ANCHORS = {"continuous": engine.ANCHOR_CONTINUOUS, "every-chunk": engine.ANCHOR_EVERY_CHUNK,
           "every-n": engine.ANCHOR_EVERY_N}
FRAMINGS = {"face": engine.FRAMING_FACE, "whole": engine.FRAMING_WHOLE}
POD_MODELS = "/workspace/runpod-slim/ComfyUI/models/flashhead"


def load_audio(path, sample_rate=48000):
    p = subprocess.run([engine.ffmpeg(), "-v", "error", "-i", path, "-vn", "-f", "f32le",
                        "-ac", "2", "-ar", str(sample_rate), "-"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=True)
    data = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2).T.copy()
    return torch.from_numpy(data), sample_rate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portrait", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor", choices=ANCHORS, default="continuous")
    ap.add_argument("--reset-seconds", type=float, default=10.0)
    ap.add_argument("--framing", choices=FRAMINGS, default="face")
    ap.add_argument("--face-ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["pro", "lite"], default="pro",
                    help="lite needs the Lite weights, which this template does not download")
    ap.add_argument("--color-correction", type=float, default=1.0)
    ap.add_argument("--models", default=os.environ.get("FLASHHEAD_MODELS", POD_MODELS))
    ap.add_argument("--upstream", default=None, help="SoulX-FlashHead checkout")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile (needed where Triton is unavailable)")
    ap.add_argument("--seconds", type=float, default=0,
                    help="render only the first N seconds of the audio")
    args = ap.parse_args()
    if args.upstream:
        os.environ["FLASHHEAD_ROOT"] = os.path.abspath(args.upstream)

    from PIL import Image
    rgb = np.asarray(Image.open(args.portrait).convert("RGB"))
    portrait = engine.frame_portrait(rgb, FRAMINGS[args.framing], args.face_ratio)
    Image.fromarray(portrait).save(os.path.splitext(args.out)[0] + "_portrait.png")

    waveform, sr = load_audio(args.audio)
    if args.seconds:
        waveform = waveform[:, :int(args.seconds * sr)]

    pipeline, motion = engine.load_pipeline(
        os.path.join(args.models, "SoulX-FlashHead-1_3B"),
        os.path.join(args.models, "wav2vec2-base-960h"),
        model_type=args.model, compile_model=not args.no_compile)
    stats = engine.render(pipeline, motion, portrait, waveform, sr, args.out,
                          anchor=ANCHORS[args.anchor], reset_seconds=args.reset_seconds,
                          seed=args.seed, color_correction=args.color_correction,
                          on_chunk=lambda i, n: print(f"  chunk {i + 1}/{n}", end="\r"))
    stats["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    stats["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) \
        if torch.cuda.is_available() else None
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
