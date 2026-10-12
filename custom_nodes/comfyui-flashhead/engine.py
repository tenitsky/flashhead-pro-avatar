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
  refresh at a pause    the same re-anchor once in every 20-40 s, in the middle of a
                        real pause in the speech (mouth shut), planned from the whole
                        voiceover; the chunk before it ends in the pause

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
ANCHOR_PAUSES = "refresh from portrait at a pause"
ANCHOR_EVERY_CHUNK = "reset to portrait every chunk"
ANCHOR_EVERY_N = "reset to portrait every N seconds"
ANCHORS = [ANCHOR_CONTINUOUS, ANCHOR_PAUSES, ANCHOR_EVERY_CHUNK, ANCHOR_EVERY_N]
# Refresh at a pause: once in every reset_seconds to reset_seconds + PAUSE_WINDOW
# (20-40 s by default), in the middle of a real pause, never in the last REFRESH_TAIL s.
REFRESH_SECONDS = 20.0
PAUSE_WINDOW = 20.0
MIN_PAUSE = 0.4             # seconds of near-silence that count as a real pause
PAUSE_SEARCH_EXTRA = 20.0   # look this much past the window for a real pause
REFRESH_TAIL = 3.0
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS   # 640

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


def frame_loudness(mono16, total):
    """Loudness of every video frame in dB (RMS of its 640 samples at 16 kHz)."""
    rms = np.zeros(total, dtype=np.float64)
    usable = min(total, len(mono16) // SAMPLES_PER_FRAME)
    if usable:
        audio = mono16[:usable * SAMPLES_PER_FRAME].astype(np.float64)
        rms[:usable] = np.sqrt((audio.reshape(usable, SAMPLES_PER_FRAME) ** 2).mean(1))
    return 20 * np.log10(rms + 1e-9)


def find_pauses(mono16, total):
    """Every stretch of near-silence as (first_frame, end_frame), end exclusive.

    The threshold adapts to the recording: 30% of the way from its noise floor (5th
    percentile) to its speech level (95th), and at least 20 dB under the speech, so
    room tone counts as silence and soft syllables do not. A single louder frame
    inside a pause (a click, a breath) does not split it.
    """
    if total == 0:
        return []
    db = frame_loudness(mono16, total)
    floor, speech = np.percentile(db, 5), np.percentile(db, 95)
    threshold = min(floor + 0.3 * (speech - floor), speech - 20.0)
    quiet = db < threshold
    quiet[1:-1] |= quiet[:-2] & quiet[2:]          # bridge one-frame bumps
    pauses, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            pauses.append((start, i))
            start = None
    if start is not None:
        pauses.append((start, total))
    return pauses


def pause_refreshes(mono16, total, min_seconds=REFRESH_SECONDS, window=PAUSE_WINDOW,
                    min_pause=MIN_PAUSE, details=False):
    """Frames where the render starts again from the portrait, planned from the whole
    voiceover before any chunk renders.

    Walking from the start, each refresh goes in the middle of the longest real pause
    (at least min_pause seconds of near-silence, mouth shut) between min_seconds and
    min_seconds + window after the previous one. With no such pause there, the first
    real pause in the next PAUSE_SEARCH_EXTRA seconds is used instead of cutting into
    speech; only if there is none either does the longest shorter gap in the window
    get it, and as a last resort its quietest frame. details=True returns
    (frame, pause_seconds, how) tuples.
    """
    if total <= 0:
        return []
    lo = max(1.0, float(min_seconds))
    lo_f, hi_f = int(round(lo * FPS)), int(round((lo + max(0.0, window)) * FPS))
    extra_f = int(round(PAUSE_SEARCH_EXTRA * FPS))
    need = max(1, int(round(float(min_pause) * FPS)))
    db = frame_loudness(mono16, total)
    pauses = find_pauses(mono16, total)
    end_limit = total - int(round(REFRESH_TAIL * FPS))
    plan, last = [], 0

    def mid(p):
        return (p[0] + p[1]) // 2

    while True:
        first, end = last + lo_f, min(last + hi_f, end_limit)
        if first >= end:
            break
        in_window = [p for p in pauses if first <= mid(p) <= end]
        real = [p for p in in_window if p[1] - p[0] >= need]
        later = [p for p in pauses if end < mid(p) <= min(last + hi_f + extra_f, end_limit)
                 and p[1] - p[0] >= need]
        if real:
            pick, how = max(real, key=lambda p: (p[1] - p[0], -db[p[0]:p[1]].mean())), "pause"
        elif later:
            pick, how = later[0], "pause after the window"
        elif in_window:
            pick, how = max(in_window, key=lambda p: p[1] - p[0]), "short gap"
        else:
            frame = first + int(np.argmin(db[first:end + 1]))
            pick, how = (frame, frame + 1), "no gap: quietest frame"
        last = mid(pick)
        plan.append((last, (pick[1] - pick[0]) / FPS, how))
    return plan if details else [f for f, _, _ in plan]


def refresh_plan(anchor, reset_seconds, new_frames, total, mono16):
    """Every (frame, pause_seconds, how) where the render starts again from the
    portrait (frame 0 not counted). Fixed schedules restart on chunk boundaries."""
    if anchor == ANCHOR_PAUSES:
        return pause_refreshes(mono16, total, float(reset_seconds), details=True)
    interval = reset_interval(anchor, reset_seconds, new_frames)
    chunks = max(1, math.ceil(total / new_frames))
    return [(i * new_frames, 0.0, "schedule") for i in range(1, chunks)
            if resets_before(i, interval)]


def resets_before(chunk, interval):
    """Whether chunk `chunk` (0-based) starts from the portrait because of a reset.
    Chunk 0 always starts from the portrait; it is not counted as a reset."""
    return interval > 0 and chunk > 0 and chunk % interval == 0


def schedule(total, new_frames, refreshes=()):
    """Chunks as (first_frame, frames_kept, starts_from_portrait).

    Each chunk renders new_frames frames. The chunk before a refresh keeps only the
    frames up to it, so the refresh starts exactly there, in the middle of the pause;
    the last chunk keeps all its frames (the mux cuts the video to the audio).
    """
    bounds = sorted({int(f) for f in refreshes if 0 < f < total})
    chunks, start, from_portrait = [], 0, False
    for end in bounds + [total]:
        while True:
            keep = new_frames if end == total else min(new_frames, end - start)
            chunks.append((start, keep, from_portrait))
            start, from_portrait = start + keep, False
            if start >= end:
                break
        from_portrait = True
    return chunks


def video_frames(samples):
    """Video frames covering a 16 kHz track of `samples` samples."""
    return max(1, math.ceil(samples / SAMPLES_PER_FRAME))


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
    total = video_frames(mono.size)
    refreshes = refresh_plan(anchor, reset_seconds, new_frames, total, mono)
    plan = schedule(total, new_frames, [f for f, _, _ in refreshes])
    chunks = len(plan)
    cached = SAMPLE_RATE * CACHED_AUDIO_SECONDS
    # Zeros before the audio (the rolling window starts silent, as upstream) and after.
    padded = np.zeros(cached + (plan[-1][0] + new_frames) * SAMPLES_PER_FRAME, np.float32)
    padded[cached:cached + mono.size] = mono

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

        end = CACHED_AUDIO_SECONDS * FPS
        start = end - FRAME_NUM
        writer = VideoWriter(os.path.join(work, "video.mp4"))
        chunk_seconds, resets = [], 0
        log(f"{LOG} {mono.size / SAMPLE_RATE:.1f} s of audio -> {chunks} chunks of "
            f"up to {new_frames} frames, anchor: {anchor}"
            + (f", {len(refreshes)} restarts from the portrait" if refreshes else ""))
        if anchor == ANCHOR_PAUSES:
            for frame, pause, how in refreshes:
                log(f"{LOG}   restart at {frame / FPS:.2f} s ({how}, {pause:.2f} s)")
        try:
            for i, (first, keep, from_portrait) in enumerate(plan):
                t0 = time.time()
                if from_portrait:
                    pipeline.reset_person_name()
                    resets += 1
                # The 8 s of audio ending where this chunk's new frames end.
                stop = cached + (first + new_frames) * SAMPLES_PER_FRAME
                emb = audio_embedding(pipeline, padded[stop - cached:stop].copy(),
                                      start, end)
                with contextlib.redirect_stdout(io.StringIO()):  # per-step timings
                    sample = pipeline.generate(emb.to(pipeline.device))
                frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255)
                frames = frames[motion_frames_num:].round().to(torch.uint8).cpu().numpy()
                writer.write(frames[:keep])
                chunk_seconds.append(time.time() - t0)
                if i == 0 or (i + 1) % 25 == 0 or i + 1 == chunks:
                    log(f"{LOG} chunk {i + 1}/{chunks} "
                          f"({(first + keep) / FPS:.0f} s of video, "
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
        "output": out_path, "anchor": anchor, "resets": resets,
        "reset_times_s": [round(f / FPS, 2) for f, _, _ in refreshes],
        "reset_pauses_s": [round(p, 2) for _, p, _ in refreshes],
        "chunks": chunks, "frames_per_chunk": new_frames,
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
