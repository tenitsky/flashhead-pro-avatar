"""The pinned SoulX-FlashHead commit still matches what engine.py assumes: the pipeline
API it calls, the chunk settings it copies, the reset it uses, and that the model and
LTX VAE code import with the stubs and the pinned diffusers. No weights or GPU needed.

    python tools/test_upstream.py          # fetches the pinned commit (needs network)
    FLASHHEAD_ROOT=/path/to/checkout python tools/test_upstream.py
"""
import importlib.util
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")
REF = re.search(r'FLASHHEAD_REF="\$\{FLASHHEAD_REF:-([0-9a-f]{40})\}"', SETUP).group(1)
URL = re.search(r'FLASHHEAD_URL="([^"]+)"', SETUP).group(1)


GIVEN = bool(os.environ.get("FLASHHEAD_ROOT"))  # a local checkout, at any commit


def checkout():
    if GIVEN:
        return Path(os.environ["FLASHHEAD_ROOT"])
    target = Path(tempfile.mkdtemp(prefix="flashhead-upstream-"))
    for cmd in (["git", "init", "-q", str(target)],
                ["git", "-C", str(target), "fetch", "-q", "--depth", "1", URL, REF],
                ["git", "-C", str(target), "-c", "advice.detachedHead=false", "checkout",
                 "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, check=True)
    return target


UPSTREAM = checkout()
os.environ["FLASHHEAD_ROOT"] = str(UPSTREAM)
spec = importlib.util.spec_from_file_location(
    "engine", ROOT / "custom_nodes/comfyui-flashhead/engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
sys.path.insert(0, str(UPSTREAM))
engine.install_stubs()


def source(rel):
    return (UPSTREAM / rel).read_text(encoding="utf-8")


class UpstreamTests(unittest.TestCase):
    @unittest.skipIf(GIVEN, "testing a local checkout")
    def test_pinned_commit(self):
        head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        self.assertEqual(head, REF)

    def test_infer_params_match_engine(self):
        params = dict(re.findall(r"^(\w+):\s*([\d.]+)", source("flash_head/configs/infer_params.yaml"), re.M))
        self.assertEqual(int(params["frame_num"]), engine.FRAME_NUM)
        self.assertEqual(int(params["motion_frames_latent_num"]), engine.MOTION_LATENTS)
        self.assertEqual(int(params["tgt_fps"]), engine.FPS)
        self.assertEqual(int(params["sample_rate"]), engine.SAMPLE_RATE)
        self.assertEqual(float(params["sample_shift"]), engine.SHIFT)
        self.assertEqual(int(params["cached_audio_duration"]), engine.CACHED_AUDIO_SECONDS)
        self.assertEqual((int(params["height"]), int(params["width"])), (engine.SIZE, engine.SIZE))

    def test_inference_helpers_match_engine(self):
        inference = source("flash_head/inference.py")
        self.assertIn("infer_params['sample_steps'] = 4", inference)
        self.assertIn("motion_frames_num = (motion_frames_latent_num - 1) * "
                      "pipeline.config.vae_stride[0] + 1", inference)
        self.assertIn("indices = (torch.arange(2 * 2 + 1) - 2) * 1", inference)
        self.assertIn("sample_frames = (((sample+1)/2).permute(1,2,3,0).clip(0,1) * 255)",
                      inference)
        generate = source("generate_video.py")
        # Stream mode: an 8 s rolling window ending at the chunk, motion frames dropped.
        self.assertIn("audio_start_idx = audio_end_idx - frame_num", generate)
        self.assertIn("video = video[motion_frames_num:]", generate)

    def test_pipeline_api(self):
        import flash_head.src.pipeline.flash_head_pipeline as fhp
        for name in ("COMPILE_MODEL", "COMPILE_VAE"):
            self.assertTrue(hasattr(fhp, name), name)
        cls = fhp.FlashHeadPipeline
        self.assertEqual(
            list(inspect.signature(cls.__init__).parameters)[:6],
            ["self", "checkpoint_dir", "model_type", "wav2vec_dir", "device", "param_dtype"])
        prepare = inspect.signature(cls.prepare_params).parameters
        for name in ("cond_image_path_or_dir", "target_size", "frame_num", "motion_frames_num",
                     "sampling_steps", "seed", "shift", "color_correction_strength",
                     "use_face_crop"):
            self.assertIn(name, prepare)
        self.assertEqual(list(inspect.signature(cls.preprocess_audio).parameters),
                         ["self", "speech_array", "sr", "fps"])
        self.assertEqual(list(inspect.signature(cls.generate).parameters),
                         ["self", "audio_embedding"])
        # A reset puts the portrait back as the motion context, as at chunk 0.
        reset = inspect.getsource(cls.reset_person_name)
        self.assertIn("self.latent_motion_frames = self.ref_img_latent[:, :1].clone()", reset)
        # Pro loads these, which setup.sh downloads.
        init = inspect.getsource(cls.__init__)
        self.assertIn('"Model_Pro"', init)
        self.assertIn('"VAE_Wan/Wan2.1_VAE.pth"', init)

    def test_model_and_wan_vae_code_import_with_pinned_diffusers(self):
        import diffusers
        self.assertEqual(diffusers.__version__, re.search(r'"diffusers==([\d.]+)"', SETUP).group(1))
        import flash_head.src.modules.flash_head_model as model
        import flash_head.wan.modules as wan
        self.assertTrue(hasattr(model, "WanModelAudioProject"))
        self.assertTrue(hasattr(wan, "WanVAE"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
