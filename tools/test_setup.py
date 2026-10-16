"""Exercise real Bash download helpers and the start command without network access."""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("TEST_BASH") or ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")
HELPERS = SETUP[SETUP.index("file_ok() {"):SETUP.index("# 3. Model downloads")]


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class DownloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="flashhead-pro-download-")
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.fixture = self.directory / "fixture.safetensors"
        header = json.dumps({"data": {"dtype": "U8", "shape": [1000000], "data_offsets": [0, 1000000]}}).encode()
        header += b" " * (-len(header) % 8)
        self.fixture.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 1000000)

    def run_script(self, command):
        script = self.directory / "check.sh"
        preamble = ("set -e\n" + "PY=" + shlex.quote(Path(sys.executable).as_posix()) + "\n"
                    + "FIXTURE=" + shlex.quote(self.fixture.as_posix()) + "\n"
                    + "DEST=" + shlex.quote((self.directory / "model.safetensors").as_posix()) + "\n"
                    + "COMFYUI_PATH=" + shlex.quote((self.directory / "ComfyUI").as_posix()) + "\n"
                    + "MODELS_DIR=" + shlex.quote((self.directory / "ComfyUI/models/flashhead").as_posix()) + "\n"
                    + "STAGE=" + shlex.quote((self.directory / "stage").as_posix()) + "\n"
                    + "HF_TOKEN=''\nHF_BIN=''\nSTAGE_DIR=''\n")
        script.write_text(preamble + HELPERS + "\n" + command, encoding="utf-8", newline="\n")
        result = subprocess.run([BASH, str(script)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_valid_and_truncated_safetensors(self):
        self.run_script('file_ok "$FIXTURE"\n')
        self.fixture.write_bytes(self.fixture.read_bytes()[:-1])
        self.run_script('if file_ok "$FIXTURE"; then exit 1; fi\n')

    def test_successful_transfer_is_promoted(self):
        self.run_script('''
wget() { cp "$FIXTURE" "$DEST.part"; }
fetch_url "$DEST" https://example.invalid/model
file_ok "$DEST"
test ! -e "$DEST.part"
''')

    def test_failed_transfer_never_promotes_partial_even_if_large(self):
        self.run_script('''
wget() { cp "$FIXTURE" "$DEST.part"; return 1; }
if fetch_url "$DEST" https://example.invalid/model; then exit 1; fi
test ! -e "$DEST"
test -e "$DEST.part"
''')

    def test_success_exit_with_truncated_model_is_rejected(self):
        self.fixture.write_bytes(self.fixture.read_bytes()[:-1])
        self.run_script('''
wget() { cp "$FIXTURE" "$DEST.part"; }
if fetch_url "$DEST" https://example.invalid/model; then exit 1; fi
test ! -e "$DEST"
test -e "$DEST.part"
''')

    def test_valid_and_broken_json(self):
        config = self.directory / "config.json"
        config.write_text('{"dim": 1536}', encoding="utf-8")
        self.run_script('file_ok "$(dirname "$DEST")/config.json"\n')
        config.write_text('{"dim": 15', encoding="utf-8")
        self.run_script('if file_ok "$(dirname "$DEST")/config.json"; then exit 1; fi\n')

    def test_valid_and_truncated_pytorch_checkpoint(self):
        import torch
        checkpoint = self.directory / "Wan2.1_VAE.pth"
        torch.save({"weight": torch.zeros(300000)}, checkpoint)  # ~1.2 MB zip
        self.run_script('file_ok "$(dirname "$DEST")/Wan2.1_VAE.pth"\n')
        checkpoint.write_bytes(checkpoint.read_bytes()[:-100])
        self.run_script('if file_ok "$(dirname "$DEST")/Wan2.1_VAE.pth"; then exit 1; fi\n')
        checkpoint.write_bytes(b"\0" * 2000000)
        self.run_script('if file_ok "$(dirname "$DEST")/Wan2.1_VAE.pth"; then exit 1; fi\n')

    def test_repo_subfolder_is_kept_under_its_models_folder(self):
        self.run_script('''
fetch_url() { printf '%s\\n' "$2" >> "$DEST.urls"; cp "$FIXTURE" "$1"; }
download_hf Soul-AILab/SoulX-FlashHead-1_3B Model_Pro/diffusion_pytorch_model.safetensors SoulX-FlashHead-1_3B
file_ok "$MODELS_DIR/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors"
test "$(cat "$DEST.urls")" = https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B/resolve/main/Model_Pro/diffusion_pytorch_model.safetensors
''')

    def test_fast_downloader_stages_nested_paths(self):
        self.run_script('''
HF_BIN=fake_hf
fake_hf() {
  test "$1" = download && test "$2" = Soul-AILab/SoulX-FlashHead-1_3B || return 1
  mkdir -p "$5/$(dirname "$3")"
  cp "$FIXTURE" "$5/$3"
}
fetch_url() { echo 'Unexpected wget call' >&2; exit 2; }
download_hf Soul-AILab/SoulX-FlashHead-1_3B Model_Pro/diffusion_pytorch_model.safetensors SoulX-FlashHead-1_3B
file_ok "$MODELS_DIR/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors"
test -z "$(ls -A "$MODELS_DIR/SoulX-FlashHead-1_3B/Model_Pro" | grep -v diffusion_pytorch_model)"
''')

    # Global Volume: staged on the container disk, copied to the final name, never
    # renamed on the volume.
    GLOBAL = '''
STAGE_DIR="$STAGE"; mkdir -p "$STAGE_DIR"
mv() { echo 'Unexpected rename' >&2; exit 2; }
wget() { while [ "$1" != -O ]; do shift; done; printf '%s\\n' "$2" > "$DEST.wget"; cat "$FIXTURE" > "$2"; }
'''

    def test_global_transfer_is_staged_and_copied(self):
        self.run_script(self.GLOBAL + '''
fetch_url "$DEST" https://example.invalid/model
file_ok "$DEST"
case "$(cat "$DEST.wget")" in "$STAGE"/*model.safetensors.part) ;; *) exit 3 ;; esac
test -z "$(ls -A "$STAGE")"
test ! -e "$DEST.part"
''')

    def test_global_staging_names_keep_same_named_files_apart(self):
        # Both repos have a config.json: their staged copies must not share a name.
        self.run_script(self.GLOBAL + '''
wget() { while [ "$1" != -O ]; do shift; done; printf '%s\\n' "$2" >> "$DEST.wget"; return 1; }
fetch_url "$MODELS_DIR/a/config.json" https://example.invalid/a || true
fetch_url "$MODELS_DIR/b/config.json" https://example.invalid/b || true
test "$(sort -u "$DEST.wget" | wc -l)" = 2
''')

    def test_global_failed_transfer_keeps_staged_partial_for_resume(self):
        self.run_script(self.GLOBAL + '''
wget() { while [ "$1" != -O ]; do shift; done; head -c 5000 "$FIXTURE" > "$2"; return 1; }
if fetch_url "$DEST" https://example.invalid/model; then exit 1; fi
test ! -e "$DEST"
test -n "$(ls "$STAGE"/*model.safetensors.part)"
''')

    def test_global_copy_that_does_not_verify_is_not_accepted(self):
        self.run_script(self.GLOBAL + '''
cp() { command cp "$@"; truncate -s -1 "${@: -1}"; }
if fetch_url "$DEST" https://example.invalid/model; then exit 1; fi
if file_ok "$DEST"; then exit 1; fi
file_ok "$(ls "$STAGE"/*model.safetensors.part)"
''')

    def test_global_fast_downloader_stages_on_container_disk(self):
        self.run_script(self.GLOBAL + '''
HF_BIN=fake_hf
fake_hf() {
  case "$5" in "$STAGE"/*) ;; *) echo "staged outside the container disk: $5" >&2; return 1 ;; esac
  mkdir -p "$5/$(dirname "$3")"
  cat "$FIXTURE" > "$5/$3"
}
fetch_url() { echo 'Unexpected wget call' >&2; exit 2; }
download_hf Soul-AILab/SoulX-FlashHead-1_3B Model_Pro/diffusion_pytorch_model.safetensors SoulX-FlashHead-1_3B
file_ok "$MODELS_DIR/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors"
test -z "$(ls -A "$STAGE")"
''')

    def test_failed_download_keeps_original_file(self):
        self.run_script('''
mkdir -p "$MODELS_DIR/wav2vec2-base-960h"
printf 'original user file' > "$MODELS_DIR/wav2vec2-base-960h/model.safetensors"
wget() { return 1; }
if download_hf facebook/wav2vec2-base-960h model.safetensors wav2vec2-base-960h; then exit 1; fi
test "$(cat "$MODELS_DIR/wav2vec2-base-960h/model.safetensors")" = 'original user file'
''')

    def test_existing_valid_files_skip_all_downloads(self):
        self.run_script('''
mkdir -p "$MODELS_DIR/wav2vec2-base-960h"
cp "$FIXTURE" "$MODELS_DIR/wav2vec2-base-960h/model.safetensors"
printf '{"a": 1}' > "$MODELS_DIR/wav2vec2-base-960h/config.json"
hf_fast() { echo 'Unexpected HF call' >&2; exit 2; }
fetch_url() { echo 'Unexpected wget call' >&2; exit 2; }
download_hf facebook/wav2vec2-base-960h model.safetensors wav2vec2-base-960h
download_hf facebook/wav2vec2-base-960h config.json wav2vec2-base-960h
''')

    def test_bootstrap_command_matches_readme_and_parses(self):
        command_file = (ROOT / "runpod-start.json").read_text(encoding="utf-8")
        config = json.loads(command_file)
        self.assertIn(command_file.strip(), (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(config["entrypoint"], ["bash", "-lc"])
        self.assertEqual(config["cmd"][0], (ROOT / "bootstrap.sh").read_text(encoding="utf-8"))
        self.assertIn("https://github.com/tenitsky/flashhead-pro-avatar.git", config["cmd"][0])
        script = self.directory / "bootstrap.sh"
        script.write_text(config["cmd"][0], encoding="utf-8")
        subprocess.run([BASH, "-n", str(script)], check=True)


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class BootstrapTests(unittest.TestCase):
    def run_bootstrap(self, mode):
        with tempfile.TemporaryDirectory(prefix="flashhead-pro-bootstrap-") as directory:
            base = Path(directory)
            resolver = base / "resolv.conf"
            original = "nameserver 10.0.0.2\nsearch private.example\n"
            resolver.write_text(original, encoding="utf-8", newline="\n")
            fixture = base / "fixture-setup.sh"
            fixture.write_text('printf ready > "$BOOT_READY"\n', encoding="utf-8", newline="\n")
            env = {"DNS_MODE": mode, "RESOLVER": resolver.as_posix(),
                   "DNS_COUNT": (base / "dns-count").as_posix(),
                   "GIT_COUNT": (base / "git-count").as_posix(),
                   "BOOT_READY": (base / "ready").as_posix(),
                   "FIXTURE_SETUP": fixture.as_posix()}
            preamble = "\n".join("export " + key + "=" + shlex.quote(value) for key, value in env.items()) + "\n"
            preamble += '''
timeout() { shift; "$@"; }
sleep() { :; }
getent() {
  printf 'lookup\\n' >> "$DNS_COUNT"
  case "$DNS_MODE" in
    healthy|clone_failure) return 0 ;;
    delayed) test "$(wc -l < "$DNS_COUNT")" -gt 1 ;;
    recover) grep -q 'nameserver 1.1.1.1' "$RESOLVER" ;;
    failed) return 1 ;;
  esac
}
git() {
  printf 'clone\\n' >> "$GIT_COUNT"
  [ "$DNS_MODE" != clone_failure ] || return 1
  cp "$FIXTURE_SETUP" "${@: -1}/setup.sh"
}
'''
            # Run the shipped command, redirecting only filesystem paths into the
            # fixture. Resolver, timeout, sleep and Git results are simulated.
            command = json.loads((ROOT / "runpod-start.json").read_text(encoding="utf-8"))["cmd"][0]
            command = command.replace("/etc/resolv.conf", shlex.quote(resolver.as_posix()))
            command = command.replace("/tmp/flashhead-pro-avatar-dns.XXXXXX", shlex.quote((base / "dns-backup.XXXXXX").as_posix()))
            command = command.replace("/tmp/flashhead-pro-avatar.XXXXXX", shlex.quote((base / "repo.XXXXXX").as_posix()))
            script = base / "bootstrap-test.sh"
            script.write_text(preamble + command, encoding="utf-8", newline="\n")
            result = subprocess.run([BASH, str(script)], text=True, capture_output=True, timeout=15)
            backups = [path.read_text() for path in base.glob("dns-backup.*")]
            clones = (base / "git-count").read_text().splitlines() if (base / "git-count").exists() else []
            return result, resolver.read_text(), original, backups, len(clones), (base / "ready").exists()

    def test_healthy_and_transient_dns_preserve_original_resolver(self):
        for mode in ("healthy", "delayed"):
            with self.subTest(mode=mode):
                result, resolver, original, backups, clones, ready = self.run_bootstrap(mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(resolver, original)
                self.assertEqual(backups, [])
                self.assertEqual(clones, 1)
                self.assertTrue(ready)

    def test_broken_dns_recovers_before_setup(self):
        result, resolver, original, backups, clones, ready = self.run_bootstrap("recover")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nameserver 1.1.1.1", resolver)
        self.assertEqual(backups, [original])
        self.assertEqual(clones, 1)
        self.assertTrue(ready)

    def test_failed_dns_restores_original_and_never_starts_setup(self):
        result, resolver, original, backups, clones, ready = self.run_bootstrap("failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FATAL: DNS still unavailable", result.stderr)
        self.assertEqual(resolver, original)
        self.assertEqual(backups, [original])
        self.assertEqual(clones, 0)
        self.assertFalse(ready)

    def test_clone_failure_retries_without_changing_working_dns(self):
        result, resolver, original, backups, clones, ready = self.run_bootstrap("clone_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unable to clone flashhead-pro-avatar after 10 attempts", result.stderr)
        self.assertEqual(resolver, original)
        self.assertEqual(backups, [])
        self.assertEqual(clones, 10)
        self.assertFalse(ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)
