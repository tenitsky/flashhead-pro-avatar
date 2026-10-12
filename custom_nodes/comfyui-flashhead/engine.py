"""
FlashHead long-form engine: one portrait + a voiceover of any length -> one mp4.

No ComfyUI imports, so the same code runs inside the nodes and from
tools/render_local.py.

How SoulX-FlashHead builds a long video (upstream generate_video.py, stream mode):
every chunk is 33 frames at 25 fps and 512 x 512. Its first latents are overwritten
with "motion frames", the previous chunk's last frames re-encoded by the VAE, and the
portrait is passed to every chunk as the identity reference. The motion frames are
dropped from each chunk's output. Pro's Wan VAE compresses time 4x, so 2 motion latents
are 5 frames and each chunk adds 33 - 5 = 28 new frames (1.12 s); Lite's LTX VAE (8x)
carries 9 and adds 24.
The audio goes through wav2vec2 in a rolling 8-second window that ends where the chunk
ends, exactly as upstream streams it.

The anchor setting is the experiment this template exists for:

  continuous            upstream: every chunk continues from the previous one
  reset every chunk     the motion frames go back to the portrait before every chunk,
                        so every ~1 s starts again from the portrait pose
  reset every N seconds the same, but only every N seconds: a periodic re-anchor
                        against slow drift, with one visible reset at each

A reset is upstream's own reset_person_name(): the next chunk starts the way the first
one does.
"""

import contextlib
import importlib.util
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from collections import deque

import numpy as np
import torch

LOG = "[FlashHead]"


def log(*args, **kwargs):
    """print, flushed: ComfyUI's stdout is a file on the pod."""
    print(*args, **kwargs, flush=True)
FPS = 25
SAMPLE_RATE = 16000
SIZE = 512
FRAME_NUM = 33                 # frames per chunk (infer_params.yaml)
MOTION_LATENTS = 2             # latents carried into the next chunk
CACHED_AUDIO_SECONDS = 8       # rolling wav2vec2 window
SHIFT = 5.0
STEPS = 4                      # the distilled Lite and Pro models
AUDIO_CONTEXT = 2              # wav2vec2 frames either side of each video frame

ANCHOR_CONTINUOUS = "continuous (upstream)"
ANCHOR_EVERY_CHUNK = "reset to portrait every chunk"
ANCHOR_EVERY_N = "reset to portrait every N seconds"
ANCHORS = [ANCHOR_CONTINUOUS, ANCHOR_EVERY_CHUNK, ANCHOR_EVERY_N]

FRAMING_FACE = "face crop"
FRAMING_WHOLE = "whole image (center crop)"
FRAMINGS = [FRAMING_FACE, FRAMING_WHOLE]


# ---------------------------------------------------------------------------- loading
def upstream_root():
    """The pinned SoulX-FlashHead checkout setup.sh places next to this file."""
    return os.environ.get("FLASHHEAD_ROOT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "SoulX-FlashHead")


def _unavailable(what):
    def fail(*args, **kwargs):
        raise RuntimeError(f"{what} is not installed: this template runs FlashHead on "
                           f"one GPU, which does not use it.")
    return fail


def install_stubs():
    """Stand-ins for two packages upstream imports but a single GPU never calls.

    xfuser (multi-GPU sequence parallelism) is imported at the top of the model file;
    it is only used when torch.distributed is initialised. mediapipe is imported by
    upstream's face crop, which this template replaces with OpenCV. Installing either
    would pull heavy dependencies into ComfyUI's environment. A real installation
    always wins: stubs are added only for packages that are missing.
    """
    # sys.modules first: find_spec raises on a stub, which has no __spec__.
    if "xfuser" not in sys.modules and importlib.util.find_spec("xfuser") is None:
        fail = _unavailable("xfuser (multi-GPU)")
        dist = types.ModuleType("xfuser.core.distributed")
        for name in ("get_sequence_parallel_rank", "get_sequence_parallel_world_size",
                     "get_sp_group", "init_distributed_environment",
                     "initialize_model_parallel", "get_world_group"):
            setattr(dist, name, fail)
        attention = types.ModuleType("xfuser.core.long_ctx_attention")
        attention.xFuserLongContextAttention = fail
        core = types.ModuleType("xfuser.core")
        core.distributed, core.long_ctx_attention = dist, attention
        root = types.ModuleType("xfuser")
        root.core = core
        sys.modules.update({"xfuser": root, "xfuser.core": core,
                            "xfuser.core.distributed": dist,
                            "xfuser.core.long_ctx_attention": attention})
    if "mediapipe" not in sys.modules and importlib.util.find_spec("mediapipe") is None:
        mediapipe = types.ModuleType("mediapipe")

        def missing(name):
            # AttributeError, so hasattr() and inspect keep working on the stub.
            raise AttributeError(f"mediapipe is not installed (needed for '{name}'); "
                                 f"this template crops faces with OpenCV instead.")
        mediapipe.__getattr__ = missing
        sys.modules["mediapipe"] = mediapipe


def load_pipeline(ckpt_dir, wav2vec_dir, model_type="pro", compile_model=True,
                  device="cuda"):
    """Build upstream's FlashHeadPipeline. Returns (pipeline, motion_frames_num)."""
    root = upstream_root()
    if not os.path.isdir(os.path.join(root, "flash_head")):
        raise RuntimeError(f"SoulX-FlashHead code not found at {root}. Restart the pod "
                           f"so setup.sh can install it.")
    for path, what in ((ckpt_dir, "FlashHead model"), (wav2vec_dir, "wav2vec2")):
        if not os.path.isdir(path):
            raise RuntimeError(f"{what} not found at {path}. Restart the pod so setup.sh "
                               f"can download it.")
    if root not in sys.path:
        sys.path.insert(0, root)
    install_stubs()
    import flash_head.src.pipeline.flash_head_pipeline as fhp
    # torch.compile: faster chunks after a one-off warm-up of a minute or two.
    fhp.COMPILE_MODEL = fhp.COMPILE_VAE = bool(compile_model)
    pipeline = fhp.FlashHeadPipeline(checkpoint_dir=ckpt_dir, model_type=model_type,
                                     wav2vec_dir=wav2vec_dir, device=device)
    motion_frames_num = (MOTION_LATENTS - 1) * pipeline.config.vae_stride[0] + 1
    log(f"{LOG} {model_type} model loaded from {ckpt_dir} "
          f"(compile={'on' if compile_model else 'off'}, "
          f"{FRAME_NUM - motion_frames_num} new frames per chunk)")
    return pipeline, motion_frames_num


# ---------------------------------------------------------------------------- portrait
# OpenCV's YuNet face detector (opencv_zoo, MIT licence), shipped next to this file.
# OpenCV 4.5.4+ and 5.x have FaceDetectorYN; 5.x dropped the Haar cascades.
YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "face_detection_yunet_2023mar.onnx")
DETECT_SIDE = 640  # detection runs on a downscaled copy


def face_box(rgb):
    """Most confident face as (x1, y1, x2, y2) in rgb's pixels, or None."""
    import cv2
    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_SIDE / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    detector = cv2.FaceDetectorYN.create(YUNET, "", (bgr.shape[1], bgr.shape[0]),
                                         score_threshold=0.6)
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[14])[:4] / scale
    return int(x), int(y), int(x + fw), int(y + fh)


def crop_square(img_w, img_h, box, ratio=2.0):
    """Upstream's face framing: a square ratio x the face width, the face centred
    horizontally with 55% of the square above its centre. Unlike upstream, the square
    is shifted (not clipped) to stay inside the image, so it is never stretched."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min((x2 - x1) * ratio, img_w, img_h)
    left = min(max(cx - side / 2, 0), img_w - side)
    top = min(max(cy - side * 0.55, 0), img_h - side)
    return int(round(left)), int(round(top)), int(round(side))


def frame_portrait(rgb, framing=FRAMING_FACE, ratio=2.0):
    """The 512 x 512 portrait FlashHead sees, from an RGB uint8 array."""
    from PIL import Image
    h, w = rgb.shape[:2]
    if framing == FRAMING_FACE:
        box = face_box(rgb)
        if box is None:
            log(f"{LOG} no face found; using the whole image (center crop).")
        else:
            left, top, side = crop_square(w, h, box, ratio)
            rgb = rgb[top:top + side, left:left + side]
            h, w = rgb.shape[:2]
    side = min(h, w)  # center crop to a square, then resize
    top, left = (h - side) // 2, (w - side) // 2
    square = Image.fromarray(np.ascontiguousarray(rgb[top:top + side, left:left + side]))
    return np.asarray(square.resize((SIZE, SIZE), Image.LANCZOS))


# ---------------------------------------------------------------------------- audio
def to_mono16(waveform, sample_rate):
    """[channels, samples] or [batch, channels, samples] float -> 16 kHz mono numpy."""
    import torchaudio
    wav = waveform
    if wav.dim() == 3:
        wav = wav[0]
    mono = wav.float().mean(dim=0, keepdim=True)
    if int(sample_rate) != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, int(sample_rate), SAMPLE_RATE)
    return mono[0].cpu().numpy().astype(np.float32)


def audio_embedding(pipeline, audio_array, start, end):
    """Upstream get_audio_embedding: wav2vec2 features with 2 frames either side."""
    emb = pipeline.preprocess_audio(audio_array, sr=SAMPLE_RATE, fps=FPS)
    offsets = torch.arange(2 * AUDIO_CONTEXT + 1) - AUDIO_CONTEXT
    index = torch.arange(start, end).unsqueeze(1) + offsets.unsqueeze(0)
    index = torch.clamp(index, min=0, max=end - 1)
    return emb[index][None, ...].contiguous()


# ---------------------------------------------------------------------------- planning
def reset_interval(anchor, reset_seconds, new_frames):
    """Chunks between resets to the portrait, or 0 for never."""
    if anchor == ANCHOR_EVERY_CHUNK:
        return 1
    if anchor == ANCHOR_EVERY_N:
        return max(1, round(float(reset_seconds) * FPS / new_frames))
    return 0


def resets_before(chunk, interval):
    """Whether chunk `chunk` (0-based) starts from the portrait because of a reset.
    Chunk 0 always starts from the portrait; it is not counted as a reset."""
    return interval > 0 and chunk > 0 and chunk % interval == 0


def plan(samples, new_frames):
    """(chunk count, samples per chunk) for a 16 kHz track of `samples` samples."""
    per_chunk = new_frames * SAMPLE_RATE // FPS
    return max(1, math.ceil(samples / per_chunk)), per_chunk


# ---------------------------------------------------------------------------- output
def ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found (apt-get install -y ffmpeg).")
    return exe


class VideoWriter:
    """Streams frames to ffmpeg, so minutes of video never sit in memory."""

    def __init__(self, path, crf=17):
        self.path = path
        self.proc = subprocess.Popen(
            [ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, frames):
        self.proc.stdin.write(np.ascontiguousarray(frames).tobytes())
        self.frames += len(frames)

    def close(self):
        _, err = self.proc.communicate()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing video:\n{err.decode()[-1500:]}")


def write_wav(waveform, sample_rate, path):
    wav = waveform[0] if waveform.dim() == 3 else waveform
    pcm = (wav.detach().cpu().float().numpy().T.clip(-1, 1) * 32767).astype(np.int16)
    p = subprocess.run([ffmpeg(), "-y", "-v", "error", "-f", "s16le", "-ar",
                        str(int(sample_rate)), "-ac", str(pcm.shape[1]), "-i", "-", path],
                       input=pcm.tobytes(), stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing audio:\n{p.stderr.decode()[-1500:]}")


def mux(video, wav, out):
    """The original audio over the silent video, cut to the audio's length."""
    p = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", video, "-i", wav,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed muxing audio:\n{p.stderr.decode()[-1500:]}")


# ---------------------------------------------------------------------------- render
@torch.no_grad()
def render(pipeline, motion_frames_num, portrait, waveform, sample_rate, out_path,
           anchor=ANCHOR_CONTINUOUS, reset_seconds=10.0, seed=42,
           color_correction=1.0, on_chunk=None):
    """Render the whole voiceover to out_path. Returns a stats dict.

    portrait: 512 x 512 RGB uint8 array (frame_portrait). waveform: [channels,
    samples] or [1, channels, samples] float at sample_rate. on_chunk(i, total) is
    called after every chunk and may raise to cancel.
    """
    new_frames = FRAME_NUM - motion_frames_num
    mono = to_mono16(waveform, sample_rate)
    if mono.size == 0:
        raise RuntimeError("The voiceover is empty.")
    chunks, per_chunk = plan(mono.size, new_frames)
    padded = np.zeros(chunks * per_chunk, dtype=np.float32)
    padded[:mono.size] = mono
    interval = reset_interval(anchor, reset_seconds, new_frames)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    work = tempfile.mkdtemp(prefix="flashhead-", dir=os.path.dirname(os.path.abspath(out_path)))
    try:
        from PIL import Image
        portrait_png = os.path.join(work, "portrait.png")
        Image.fromarray(portrait).save(portrait_png)
        started = time.time()
        pipeline.prepare_params(
            cond_image_path_or_dir=portrait_png, target_size=(SIZE, SIZE),
            frame_num=FRAME_NUM, motion_frames_num=motion_frames_num,
            sampling_steps=STEPS, seed=int(seed), shift=SHIFT,
            color_correction_strength=float(color_correction), use_face_crop=False)

        cached = SAMPLE_RATE * CACHED_AUDIO_SECONDS
        end = CACHED_AUDIO_SECONDS * FPS
        start = end - FRAME_NUM
        window = deque([0.0] * cached, maxlen=cached)
        writer = VideoWriter(os.path.join(work, "video.mp4"))
        chunk_seconds, resets = [], 0
        log(f"{LOG} {mono.size / SAMPLE_RATE:.1f} s of audio -> {chunks} chunks of "
              f"{new_frames} frames, anchor: {anchor}"
              + (f" (every {interval} chunks)" if interval > 1 else ""))
        try:
            for i in range(chunks):
                t0 = time.time()
                if resets_before(i, interval):
                    pipeline.reset_person_name()
                    resets += 1
                window.extend(padded[i * per_chunk:(i + 1) * per_chunk].tolist())
                emb = audio_embedding(pipeline, np.array(window, dtype=np.float32),
                                      start, end)
                with contextlib.redirect_stdout(io.StringIO()):  # per-step timings
                    sample = pipeline.generate(emb.to(pipeline.device))
                frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255)
                frames = frames[motion_frames_num:].round().to(torch.uint8).cpu().numpy()
                writer.write(frames)
                chunk_seconds.append(time.time() - t0)
                if i == 0 or (i + 1) % 25 == 0 or i + 1 == chunks:
                    log(f"{LOG} chunk {i + 1}/{chunks} "
                          f"({(i + 1) * new_frames / FPS:.0f} s of video, "
                          f"{time.time() - started:.0f} s elapsed)")
                if on_chunk:
                    on_chunk(i, chunks)
        finally:
            writer.close()
        render_seconds = time.time() - started

        wav = os.path.join(work, "audio.wav")
        write_wav(waveform, sample_rate, wav)
        mux(os.path.join(work, "video.mp4"), wav, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    seconds = mono.size / SAMPLE_RATE
    # The first chunks include torch.compile's warm-up; steady state excludes them.
    steady = chunk_seconds[3:] or chunk_seconds
    stats = {
        "output": out_path, "anchor": anchor, "reset_every_chunks": interval,
        "resets": resets, "chunks": chunks, "frames_per_chunk": new_frames,
        "audio_seconds": round(seconds, 3), "render_seconds": round(render_seconds, 2),
        "seconds_per_video_second": round(render_seconds / seconds, 3),
        "steady_chunk_seconds": round(float(np.median(steady)), 3),
        "steady_seconds_per_video_second": round(
            float(np.median(steady)) / (new_frames / FPS), 3),
        "first_chunk_seconds": round(chunk_seconds[0], 2),
    }
    log(f"{LOG} {seconds:.1f} s of video in {render_seconds:.1f} s "
          f"({stats['seconds_per_video_second']:.2f} s per video second; steady state "
          f"{stats['steady_seconds_per_video_second']:.2f}) -> {out_path}")
    return stats
