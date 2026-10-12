"""The FlashHead engine on CPU: chunk plan, audio windows, anchor resets, face crop,
stubs and the ffmpeg output. A fake pipeline stands in for the model, recording what
upstream's FlashHeadPipeline would be asked to do. python tools/test_engine.py"""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "engine", ROOT / "custom_nodes/comfyui-flashhead/engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
try:
    import cv2
    HAVE_YUNET = hasattr(cv2, "FaceDetectorYN")
except ImportError:
    HAVE_YUNET = False

MOTION = 5          # Pro: (2 - 1) * 4 + 1 (Wan VAE, 4x in time)
NEW = engine.FRAME_NUM - MOTION


class FakePipeline:
    """Records calls; returns frames whose pixel value encodes the chunk number."""
    device = "cpu"

    def __init__(self):
        self.prepared, self.windows, self.resets, self.chunk = None, [], [], 0

    def prepare_params(self, **kwargs):
        self.prepared = kwargs

    def reset_person_name(self, person_name=None):
        self.resets.append(self.chunk)

    def preprocess_audio(self, speech_array, sr=16000, fps=25):
        self.windows.append(np.asarray(speech_array).copy())
        frames = int(len(speech_array) * fps / sr)
        return torch.arange(frames, dtype=torch.float32)[:, None, None].repeat(1, 2, 3)

    def generate(self, audio_embedding):
        assert audio_embedding.shape[:3] == (1, engine.FRAME_NUM, 5)
        value = (self.chunk % 200) / 100 - 1   # pixel value (v + 1) / 2 * 255
        self.chunk += 1
        return torch.full((3, engine.FRAME_NUM, 8, 8), value)


class Recorder:
    instances = []

    def __init__(self, path, crf=17):
        self.frames, self.path = [], path
        Recorder.instances.append(self)

    def write(self, frames):
        self.frames.append(frames)

    def close(self):
        pass


class RenderLoopTests(unittest.TestCase):
    def setUp(self):
        self.real = engine.VideoWriter, engine.write_wav, engine.mux
        engine.VideoWriter = Recorder
        engine.write_wav = lambda *a, **k: None
        engine.mux = lambda *a, **k: None
        Recorder.instances.clear()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.restore)
        self.portrait = np.zeros((512, 512, 3), np.uint8)

    def restore(self):
        engine.VideoWriter, engine.write_wav, engine.mux = self.real

    def run_render(self, seconds, anchor=engine.ANCHOR_CONTINUOUS, reset_seconds=10.0,
                   sample_rate=48000):
        pipe = FakePipeline()
        t = torch.arange(int(seconds * sample_rate)) / sample_rate
        wave = torch.stack([torch.sin(2 * np.pi * 220 * t)] * 2) * 0.3
        stats = engine.render(pipe, MOTION, self.portrait, wave, sample_rate,
                              str(Path(self.dir.name) / "out.mp4"), anchor=anchor,
                              reset_seconds=reset_seconds, seed=7, color_correction=0.5)
        return pipe, stats, Recorder.instances[-1].frames

    def test_upstream_settings_reach_the_pipeline(self):
        pipe, _, _ = self.run_render(2)
        p = pipe.prepared
        self.assertEqual((p["frame_num"], p["motion_frames_num"], p["sampling_steps"],
                          p["shift"], p["seed"], p["color_correction_strength"]),
                         (33, MOTION, 4, 5.0, 7, 0.5))
        self.assertEqual(p["target_size"], (512, 512))
        self.assertFalse(p["use_face_crop"])

    def test_chunks_cover_the_audio_with_28_new_frames_each(self):
        for seconds in (0.5, 0.96, 1.0, 10.0, 71.01):
            with self.subTest(seconds=seconds):
                pipe, stats, frames = self.run_render(seconds)
                chunks = int(np.ceil(round(seconds * 16000) / (NEW * 640)))
                self.assertEqual(stats["chunks"], chunks)
                self.assertEqual([len(f) for f in frames], [NEW] * chunks)
                self.assertGreaterEqual(chunks * NEW / 25, seconds)
                # The motion frames (first 5 of each chunk) never reach the video.
                self.assertTrue(all(f.dtype == np.uint8 and f.shape[1:] == (8, 8, 3)
                                    for f in frames))

    def test_audio_window_ends_with_each_chunks_audio(self):
        pipe, _, _ = self.run_render(5)
        per_chunk = NEW * 640
        mono = engine.to_mono16(
            torch.stack([torch.sin(2 * np.pi * 220 * (torch.arange(240000) / 48000))] * 2)
            * 0.3, 48000)
        padded = np.zeros(len(pipe.windows) * per_chunk, np.float32)
        padded[:mono.size] = mono
        for i, window in enumerate(pipe.windows):
            self.assertEqual(len(window), engine.SAMPLE_RATE * engine.CACHED_AUDIO_SECONDS)
            np.testing.assert_allclose(window[-per_chunk:],
                                       padded[i * per_chunk:(i + 1) * per_chunk], atol=1e-6)

    def test_anchor_modes_reset_on_schedule(self):
        cases = {
            engine.ANCHOR_CONTINUOUS: [],
            engine.ANCHOR_EVERY_CHUNK: list(range(1, 19)),
            engine.ANCHOR_EVERY_N: [9, 18],      # 10 s at 1.12 s per chunk
        }
        for anchor, expected in cases.items():
            with self.subTest(anchor=anchor):
                pipe, stats, _ = self.run_render(21, anchor=anchor)
                self.assertEqual(stats["chunks"], 19)
                self.assertEqual(pipe.resets, expected)
                self.assertEqual(stats["resets"], len(expected))

    def test_reset_interval(self):
        self.assertEqual(engine.reset_interval(engine.ANCHOR_CONTINUOUS, 10, NEW), 0)
        self.assertEqual(engine.reset_interval(engine.ANCHOR_EVERY_CHUNK, 10, NEW), 1)
        self.assertEqual(engine.reset_interval(engine.ANCHOR_EVERY_N, 10, NEW), 9)
        self.assertEqual(engine.reset_interval(engine.ANCHOR_EVERY_N, 0.1, NEW), 1)
        self.assertFalse(engine.resets_before(0, 1))

    def test_stats_report_rendering_speed(self):
        _, stats, _ = self.run_render(3)
        for key in ("render_seconds", "seconds_per_video_second", "steady_chunk_seconds",
                    "first_chunk_seconds"):
            self.assertGreaterEqual(stats[key], 0)
        self.assertEqual(stats["audio_seconds"], 3.0)

    def test_empty_audio_is_refused(self):
        with self.assertRaises(RuntimeError):
            engine.render(FakePipeline(), MOTION, self.portrait, torch.zeros(2, 0), 48000,
                          str(Path(self.dir.name) / "x.mp4"))


class PortraitTests(unittest.TestCase):
    def test_crop_is_square_inside_and_upstream_framed(self):
        left, top, side = engine.crop_square(2048, 2048, (800, 500, 1200, 900), ratio=2.0)
        self.assertEqual(side, 800)
        self.assertEqual((left, top), (600, 700 - 440))  # centre x, 55% above centre
        for box in ((0, 0, 300, 300), (1900, 1900, 2048, 2048), (100, 900, 1900, 1700)):
            left, top, side = engine.crop_square(2048, 1500, box, ratio=2.5)
            self.assertTrue(0 <= left and 0 <= top and left + side <= 2048
                            and top + side <= 1500, box)

    def test_whole_image_is_center_cropped_to_512(self):
        rgb = np.zeros((600, 1000, 3), np.uint8)
        rgb[:, 200:800] = 255
        out = engine.frame_portrait(rgb, engine.FRAMING_WHOLE)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertGreater(out.mean(), 250)  # the white centre square, not the edges

    @unittest.skipUnless(HAVE_YUNET, "OpenCV with FaceDetectorYN required")
    def test_face_found_in_the_sample_portrait(self):
        from PIL import Image
        rgb = np.asarray(Image.open(ROOT / "samples/sample_portrait.png").convert("RGB"))
        x1, y1, x2, y2 = engine.face_box(rgb)
        h, w = rgb.shape[:2]
        self.assertTrue(0.3 * w < (x1 + x2) / 2 < 0.7 * w)    # face roughly centred
        self.assertTrue(0.1 * h < (y1 + y2) / 2 < 0.5 * h)    # in the upper half
        self.assertTrue(0.1 * w < x2 - x1 < 0.5 * w)
        self.assertEqual(engine.frame_portrait(rgb).shape, (512, 512, 3))

    @unittest.skipUnless(HAVE_YUNET, "OpenCV with FaceDetectorYN required")
    def test_no_face_falls_back_to_whole_image(self):
        rgb = np.full((300, 400, 3), 128, np.uint8)
        self.assertIsNone(engine.face_box(rgb))
        self.assertEqual(engine.frame_portrait(rgb).shape, (512, 512, 3))


class StubTests(unittest.TestCase):
    def test_stubs_import_and_fail_only_when_used(self):
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k.split(".")[0] in ("xfuser", "mediapipe")}
        self.addCleanup(lambda: [sys.modules.pop(k, None) for k in list(sys.modules)
                                 if k.split(".")[0] in ("xfuser", "mediapipe")]
                        or sys.modules.update(saved))
        if importlib.util.find_spec("xfuser") or importlib.util.find_spec("mediapipe"):
            self.skipTest("a real xfuser or mediapipe is installed")
        engine.install_stubs()
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention  # noqa
        import mediapipe
        with self.assertRaises(RuntimeError):
            get_sequence_parallel_world_size()
        self.assertFalse(hasattr(mediapipe, "solutions"))
        engine.install_stubs()  # idempotent


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe required")
class OutputTests(unittest.TestCase):
    def test_video_is_cut_to_the_audio_and_carries_it(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            writer = engine.VideoWriter(str(Path(d) / "v.mp4"))
            for i in range(3):
                writer.write(np.full((NEW, 512, 512, 3), 60 * i, np.uint8))
            writer.close()
            self.assertEqual(writer.frames, 3 * NEW)
            wave = torch.zeros(2, int(2.5 * 48000))
            engine.write_wav(wave, 48000, str(Path(d) / "a.wav"))
            engine.mux(str(Path(d) / "v.mp4"), str(Path(d) / "a.wav"), str(Path(d) / "o.mp4"))
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                 "-show_entries", "format=duration", "-of", "default=nw=1",
                 str(Path(d) / "o.mp4")], capture_output=True, text=True).stdout
            self.assertIn("codec_type=video", probe)
            self.assertIn("codec_type=audio", probe)
            duration = float(probe.split("duration=")[1].split()[0])
            self.assertAlmostEqual(duration, 2.5, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
