"""Check the Jupyter auth switch and Jupyter with persistent data and private runtime."""
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("TEST_BASH") or ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")
AUTH = SETUP[SETUP.index("# Apply before the image starts Jupyter"):SETUP.index('echo "=== Ensuring System Dependencies')]
# Network Volume layout: setup sets STATE_DIR=/workspace before this block.
RUNTIME = "STATE_DIR=/workspace" + chr(10) + SETUP[SETUP.index("export JUPYTER_CONFIG_DIR="):SETUP.index("# SoulX-FlashHead has no ComfyUI nodes of its own")]


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class JupyterTests(unittest.TestCase):
    # The image's start_jupyter() reads these; see runpod/containers
    # official-templates/comfyui/scripts/start.sh.
    IMAGE_START = 'if [ "${JUPYTER_DISABLE_AUTH:-}" = "true" ]; then JUPYTER_TOKEN=""; fi\n'

    def run_auth(self, start_text, env):
        with tempfile.TemporaryDirectory(prefix="flashhead-auth-") as directory:
            base = Path(directory)
            start = base / "start.sh"
            start.write_text(start_text, encoding="utf-8")
            script = base / "auth.sh"
            script.write_text("set -e\n" + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
                              + AUTH.replace("/start.sh", shlex.quote(start.as_posix()))
                              + 'printf "DISABLE=%s\\n" "${JUPYTER_DISABLE_AUTH-unset}"\n',
                              encoding="utf-8", newline="\n")
            result = subprocess.run([BASH, str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(start.read_text(), start_text, "setup must not edit the image's start script")
            return result.stdout

    def test_no_auth_uses_image_switch_without_editing_start_script(self):
        for env in ({}, {"JUPYTER_NO_AUTH": "1"}):
            with self.subTest(env=env):
                out = self.run_auth(self.IMAGE_START, env)
                self.assertIn("DISABLE=true", out)
                self.assertNotIn("WARNING", out)

    def test_authenticated_mode_and_unknown_image_never_stop_setup(self):
        out = self.run_auth(self.IMAGE_START, {"JUPYTER_NO_AUTH": "0", "JUPYTER_PASSWORD": "secret",
                                               "JUPYTER_DISABLE_AUTH": "true"})
        self.assertIn("DISABLE=unset", out)
        self.assertNotIn("NOTE", out)
        out = self.run_auth(self.IMAGE_START, {"JUPYTER_NO_AUTH": "0"})
        self.assertIn("does not start JupyterLab", out)
        out = self.run_auth("unsupported startup format\n", {"JUPYTER_NO_AUTH": "1"})
        self.assertIn("DISABLE=true", out)
        self.assertIn("has no JUPYTER_DISABLE_AUTH switch", out)

    @unittest.skipUnless(os.name == "posix", "Linux runtime permissions and server startup")
    def test_private_runtime_starts_real_jupyter_and_keeps_user_data_persistent(self):
        # The shipped setup block uses real secure_write, then the actual Jupyter
        # server writes its own cookie and server-info files in that directory.
        with tempfile.TemporaryDirectory(prefix="flashhead-jupyter-") as directory:
            base = Path(directory)
            workspace = base / "workspace"
            workspace.mkdir()
            script = base / "runtime.sh"
            script.write_text(
                "set -e\npython3() { " + shlex.quote(sys.executable) + ' "$@"; }\n'
                + RUNTIME.replace("/workspace", workspace.as_posix())
                + "python3 - <<'PY'\nimport json, os\nprint(json.dumps({k: os.environ[k] for k in "
                  "('JUPYTER_CONFIG_DIR', 'JUPYTER_DATA_DIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR')}))\nPY\n",
                encoding="utf-8", newline="\n")
            result = subprocess.run([BASH, str(script)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            env = json.loads(result.stdout.splitlines()[-1])
            runtime = Path(env["JUPYTER_RUNTIME_DIR"])
            self.addCleanup(shutil.rmtree, runtime)
            self.assertNotIn(workspace, runtime.parents)
            self.assertEqual(runtime.stat().st_mode & 0o777, 0o700)
            for key in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "IPYTHONDIR"):
                self.assertIn(workspace, Path(env[key]).parents)
            self.assertFalse((runtime / "secure-write-check").exists())

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log_path = base / "jupyter.log"
            with log_path.open("w", encoding="utf-8") as log:
                server = subprocess.Popen([
                    sys.executable, "-m", "jupyter_server", "--allow-root", "--no-browser",
                    "--ip=127.0.0.1", f"--port={port}", "--ServerApp.port_retries=0",
                    f"--ServerApp.root_dir={workspace}", '--ServerApp.jpserver_extensions={}',
                    "--IdentityProvider.token=", "--PasswordIdentityProvider.hashed_password=",
                ], env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 30
                    ready = False
                    while time.monotonic() < deadline and server.poll() is None:
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api", timeout=1) as response:
                                ready = response.status == 200 and "version" in json.load(response)
                            if ready:
                                break
                        except (OSError, urllib.error.URLError):
                            time.sleep(0.2)
                    self.assertTrue(ready, log_path.read_text())
                    private_files = list(runtime.glob("jpserver-*.json"))
                    self.assertTrue(private_files, log_path.read_text())
                    for path in private_files:
                        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
