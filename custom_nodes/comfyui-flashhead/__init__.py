"""
FlashHead Pro - SoulX-FlashHead (Soul-AILab, 1.3B) talking heads of any length in
ComfyUI, with the higher-quality Pro model.

FlashHead renders fast enough that one queue run renders a whole voiceover: the Render node loops over every ~1 s chunk itself and streams the frames to
ffmpeg. The model code is upstream's, pinned and installed by setup.sh next to this
file; engine.py drives it. See engine.py for how chunks join and what the anchor
setting does.
"""

import json
import math
import os
import re

import numpy as np
import torch

import folder_paths

from . import engine

LOG = engine.LOG
CATEGORY = "FlashHead"
MODELS = ["pro"]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")
STATS_DIR = "flashhead_stats"

_loaded = {}  # (model, compile) -> (pipeline, motion_frames_num)


def models_dir():
    return os.environ.get("FLASHHEAD_MODELS") or os.path.join(folder_paths.models_dir,
                                                              "flashhead")


def _progress(total):
    try:
        import comfy.utils
        bar = comfy.utils.ProgressBar(total)
    except Exception:  # pragma: no cover - outside ComfyUI
        bar = None

    def step(i, n):
        try:
            import comfy.model_management
            comfy.model_management.throw_exception_if_processing_interrupted()
        except ImportError:  # pragma: no cover
            pass
        if bar:
            bar.update_absolute(i + 1, n)
    return step


def _audio_name(prompt, default="flashhead"):
    """The Load Audio selection, without folder or extension."""
    if isinstance(prompt, dict):
        for node in prompt.values():
            if isinstance(node, dict) and node.get("class_type") == "LoadAudio":
                value = (node.get("inputs") or {}).get("audio")
                if isinstance(value, str) and value.strip():
                    stem = os.path.splitext(os.path.basename(value.replace("\\", "/")))[0]
                    if stem.strip():
                        return stem.strip()
    return default


def _natural_key(name):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def _portrait_array(image):
    """ComfyUI IMAGE (first of the batch) -> RGB uint8 array."""
    return (image[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def _render(flashhead, portrait, waveform, sample_rate, name, anchor, reset_seconds,
            seed, color_correction):
    pipeline, motion = flashhead
    out = os.path.join(folder_paths.get_output_directory(), name + ".mp4")
    # An estimate for the progress bar; restarts add a chunk each, and every update
    # carries the real total.
    chunks = math.ceil(engine.video_frames(engine.to_mono16(waveform, sample_rate).size)
                       / (engine.FRAME_NUM - motion))
    stats = engine.render(pipeline, motion, _portrait_array(portrait), waveform,
                          sample_rate, out, anchor=anchor, reset_seconds=reset_seconds,
                          seed=seed, color_correction=color_correction,
                          on_chunk=_progress(chunks))
    stats["peak_vram_gb"] = (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                             if torch.cuda.is_available() else None)
    stats_dir = os.path.join(folder_paths.get_output_directory(), STATS_DIR)
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    return out, stats


def _video_ui(path):
    return {"images": [{"filename": os.path.basename(path), "subfolder": "",
                        "type": "output"}], "animated": (True,)}


ANCHOR_TOOLTIP = ("continuous: every ~1 s chunk continues from the previous one "
                  "(upstream). refresh at a pause: continuous, but once in every 20-40 s "
                  "(reset_seconds to reset_seconds + 20 s) the render starts again from "
                  "the portrait in the middle of a real pause (0.4 s+ of silence), "
                  "planned from the whole voiceover. reset every chunk / every N "
                  "seconds: restarts from the portrait on a fixed schedule.")


class FlashHeadLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (MODELS, {"tooltip": "Pro: 512 x 512, decoded with the Wan 2.1 "
                                          "VAE; about 2.3 s per video second on an "
                                          "RTX 4090 (upstream)."}),
            "compile": ("BOOLEAN", {
                "default": True,
                "tooltip": "torch.compile: faster chunks after a one-off warm-up of a "
                           "minute or two on the first run. Turn off if it errors."}),
        }}

    RETURN_TYPES = ("FLASHHEAD",)
    RETURN_NAMES = ("flashhead",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model="pro", compile=True):
        key = (model, bool(compile))
        if key not in _loaded:
            _loaded.clear()  # one model in VRAM at a time
            torch.cuda.empty_cache()
            root = models_dir()
            _loaded[key] = engine.load_pipeline(
                os.path.join(root, "SoulX-FlashHead-1_3B"),
                os.path.join(root, "wav2vec2-base-960h"),
                model_type=model, compile_model=compile)
        return (_loaded[key],)


class FlashHeadPortrait:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "framing": (engine.FRAMINGS, {
                "tooltip": "face crop: a square around the face, as FlashHead was "
                           "trained (talking head). whole image: the picture "
                           "center-cropped to a square."}),
            "face_ratio": ("FLOAT", {
                "default": 2.0, "min": 1.2, "max": 4.0, "step": 0.1,
                "tooltip": "Face crop size as a multiple of the face width. Larger "
                           "shows more hair and shoulders; upstream uses 2.0."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("portrait",)
    FUNCTION = "frame"
    CATEGORY = CATEGORY

    def frame(self, image, framing=engine.FRAMING_FACE, face_ratio=2.0):
        square = engine.frame_portrait(_portrait_array(image), framing, face_ratio)
        return (torch.from_numpy(square.astype(np.float32) / 255.0).unsqueeze(0),)


class FlashHeadRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flashhead": ("FLASHHEAD",),
                "portrait": ("IMAGE", {"tooltip": "From FlashHead Portrait (512 x 512)."}),
                "audio": ("AUDIO",),
                "anchor": (engine.ANCHORS, {"tooltip": ANCHOR_TOOLTIP}),
                "reset_seconds": ("FLOAT", {
                    "default": 20.0, "min": 1.0, "max": 600.0, "step": 0.5,
                    "tooltip": "Refresh at a pause: earliest refresh after the last one "
                               "(the longest real pause in the following 20 s is used, "
                               "or the next one after that). Reset every N seconds: the "
                               "spacing."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "color_correction": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Pulls every chunk's colours toward the portrait "
                               "(upstream default 1.0) against colour drift."}),
                "name_suffix": ("STRING", {
                    "default": "_pro",
                    "tooltip": "The video is saved as output/<audio name><suffix>.mp4. "
                               "An existing file of that name is replaced."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, flashhead, portrait, audio, anchor, reset_seconds=10.0, seed=42,
            color_correction=1.0, name_suffix="", prompt=None):
        name = _audio_name(prompt) + name_suffix
        out, _ = _render(flashhead, portrait, audio["waveform"], audio["sample_rate"],
                         name, anchor, reset_seconds, seed, color_correction)
        return {"ui": _video_ui(out), "result": (out,)}


class FlashHeadBatchFolder:
    """Render every voiceover in a folder in one run, one video per file."""

    @classmethod
    def INPUT_TYPES(cls):
        required = dict(FlashHeadRender.INPUT_TYPES()["required"])
        del required["audio"]
        required["folder"] = ("STRING", {
            "default": "batch_audio",
            "tooltip": "A folder in ComfyUI/input/ (or an absolute path). Files whose "
                       "video already exists are skipped."})
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # depends on the folder's contents

    def run(self, flashhead, portrait, anchor, folder="batch_audio", reset_seconds=10.0,
            seed=42, color_correction=1.0, name_suffix=""):
        folder = folder.strip() or "batch_audio"
        path = folder if os.path.isabs(folder) else os.path.join(
            folder_paths.get_input_directory(), folder)
        os.makedirs(path, exist_ok=True)
        files = sorted((f for f in os.listdir(path) if f.lower().endswith(AUDIO_EXTENSIONS)
                        and not f.startswith(".")), key=_natural_key)
        if not files:
            raise RuntimeError(f"No audio files in {path}.")
        done, skipped, last = [], [], None
        for number, filename in enumerate(files, 1):
            name = os.path.splitext(filename)[0] + name_suffix
            if os.path.exists(os.path.join(folder_paths.get_output_directory(),
                                           name + ".mp4")):
                skipped.append(filename)
                continue
            engine.log(f"{LOG} batch {number}/{len(files)}: {filename}")
            waveform, sample_rate = _load_file(os.path.join(path, filename))
            last, _ = _render(flashhead, portrait, waveform, sample_rate, name, anchor,
                              reset_seconds, seed, color_correction)
            done.append(filename)
        summary = (f"Rendered {len(done)} of {len(files)} files"
                   + (f"; skipped {len(skipped)} already rendered" if skipped else "")
                   + ".")
        engine.log(f"{LOG} {summary}")
        result = {"result": (summary,)}
        if last:
            result["ui"] = _video_ui(last)
        return result


def _load_file(path, sample_rate=48000):
    import subprocess
    p = subprocess.run([engine.ffmpeg(), "-v", "error", "-i", path, "-vn", "-f", "f32le",
                        "-ac", "2", "-ar", str(sample_rate), "-"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    data = np.frombuffer(p.stdout, dtype=np.float32)
    if p.returncode != 0 or not data.size:
        raise RuntimeError(f"Could not read the audio in {path}:\n"
                           f"{p.stderr.decode(errors='replace')[-1500:]}")
    return torch.from_numpy(data.reshape(-1, 2).T.copy()), sample_rate


NODE_CLASS_MAPPINGS = {
    "FlashHeadLoader": FlashHeadLoader,
    "FlashHeadPortrait": FlashHeadPortrait,
    "FlashHeadRender": FlashHeadRender,
    "FlashHeadBatchFolder": FlashHeadBatchFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashHeadLoader": "FlashHead: Load Model",
    "FlashHeadPortrait": "FlashHead: Portrait (face crop)",
    "FlashHeadRender": "FlashHead: Render Long Video",
    "FlashHeadBatchFolder": "FlashHead: Render Batch Folder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
