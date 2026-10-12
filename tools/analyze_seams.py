#!/usr/bin/env python3
"""
Measure how a FlashHead video joins its chunks and whether it drifts.

    python tools/analyze_seams.py video.mp4 [portrait_512.png] [--chunk 24]

Seams: the mean absolute change between consecutive frames (0-255 scale), split into
the transitions that cross a chunk boundary (every 28 frames for Pro) and all
others. A seamless video has a boundary/normal ratio near 1; a visible jump at every
join pushes it well above 1.

Drift: each 10 s window's difference from the portrait, measured on a heavily
blurred 32 x 32 copy (pose, framing and colour, not mouth detail), and its mean
colour. Values climbing over time mean the video is wandering away from the
portrait. Needs only numpy, Pillow and ffmpeg.
"""
import argparse
import json
import shutil
import subprocess

import numpy as np

SIZE = 512


def frames(path, size=SIZE):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-s", f"{size}x{size}", "-"], stdout=subprocess.PIPE,
                         check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def coarse(x):
    """32 x 32 block means: pose, framing and colour without fine detail."""
    n = x.shape[-3] // 32
    return x.reshape(*x.shape[:-3], 32, n, 32, n, 3).mean(axis=(-4, -2), dtype=np.float64)


def analyze(path, portrait=None, chunk=28, window_s=10, fps=25):
    v = frames(path).astype(np.float32)
    step = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2, 3), dtype=np.float64)  # t -> t+1
    idx = np.arange(1, len(v))                                # index of the later frame
    boundary = idx % chunk == 0
    seams = {
        "frames": int(len(v)),
        "boundary_change": round(float(step[boundary].mean()), 3),
        "normal_change": round(float(step[~boundary].mean()), 3),
        "boundary_ratio": round(float(step[boundary].mean() / step[~boundary].mean()), 2),
        "worst_boundaries": [
            {"frame": int(i), "time_s": round(i / fps, 2), "change": round(float(c), 2)}
            for i, c in sorted(zip(idx[boundary], step[boundary]), key=lambda t: -t[1])[:5]],
    }
    ref = coarse(np.asarray(portrait, np.float32)) if portrait is not None else coarse(v[0])
    win = window_s * fps
    drift = []
    for start in range(0, len(v), win):
        part = v[start:start + win]
        c = coarse(part)
        drift.append({
            "from_s": start // fps,
            "structure_diff": round(float(np.abs(c - ref).mean(dtype=np.float64)), 2),
            "mean_rgb": [round(float(x), 1)
                         for x in part.mean(axis=(0, 1, 2), dtype=np.float64)],
        })
    return {"video": path, "seams": seams, "drift_per_10s": drift}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("portrait", nargs="?", help="the 512 x 512 portrait FlashHead saw")
    ap.add_argument("--chunk", type=int, default=28,
                    help="new frames per chunk: 28 for Pro, 24 for Lite")
    args = ap.parse_args()
    portrait = None
    if args.portrait:
        from PIL import Image
        portrait = np.asarray(Image.open(args.portrait).convert("RGB").resize((SIZE, SIZE)))
    print(json.dumps(analyze(args.video, portrait, args.chunk), indent=1))


if __name__ == "__main__":
    main()
