"""Network workspace checks: python tools/test_storage.py."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("storage", Path(__file__).resolve().parents[1] / "custom_nodes/comfyui-flashhead/storage.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class StorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="flashhead-storage-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.local = self.base / "workspace"
        self.newpod = self.base / "wrong-mount"
        self.local.mkdir()

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_workspace_requires_its_own_mount(self):
        for mounts in ("1 0 0:1 / / rw - overlay overlay rw\n",
                       "1 0 0:1 / / rw - ext4 /dev/root rw\n",
                       f"2 1 0:2 / {self.newpod.as_posix()} rw - nfs4 server:/data rw\n"):
            with self.subTest(mounts=mounts), patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=mounts):
                with self.assertRaisesRegex(RuntimeError, "disposable container disk"):
                    storage.require_workspace_volume(self.local)
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                storage.require_workspace_volume(self.local)

    def test_workspace_rejects_ephemeral_and_object_mounts(self):
        for kind in ("overlay", "tmpfs", "ramfs", "fuse.global", "s3fs", "fuse.geesefs", "fuse.rclone", "fuse.gcsfuse"):
            mounts = f"2 1 0:2 / {self.local.as_posix()} rw - {kind} volume rw\n"
            with self.subTest(kind=kind), patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=mounts):
                with self.assertRaisesRegex(RuntimeError, "persistent POSIX working volume"):
                    storage.require_workspace_volume(self.local)

    def test_workspace_accepts_network_and_bind_mounts(self):
        for kind in ("nfs", "nfs4", "ext4", "xfs", "fuse", "fuse.juicefs", "fuse.vstorage"):
            # Bind mounts may share the parent device; the mountinfo entry still
            # identifies the separate volume. Also exercise escaped mount paths.
            workspace = self.local / "path with spaces"
            target = workspace.as_posix().replace(" ", "\\040")
            mounts = f"2 1 0:1 /volume/subdir {target} rw - {kind} volume rw\n"
            with self.subTest(kind=kind), patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value=mounts), patch.object(storage, "probe_working_volume") as probe:
                storage.require_workspace_volume(workspace)
                storage.check_working_path(workspace / "ComfyUI/output")
                probe.assert_called_once_with(workspace.resolve())

    @unittest.skipIf(os.name == "nt", "Linux POSIX filesystem semantics required")
    def test_working_volume_probe_and_cleanup(self):
        preserved = self.write(self.local / "notebook.ipynb", b"existing user file")
        storage.probe_working_volume(self.local)
        self.assertEqual(preserved.read_bytes(), b"existing user file")
        self.assertEqual(list(self.local.iterdir()), [preserved])
        for operation in ("symlink_to", "write_bytes"):
            with self.subTest(operation=operation), patch.object(Path, operation, side_effect=OSError("not supported")):
                with self.assertRaisesRegex(RuntimeError, "failed the POSIX check"):
                    storage.probe_working_volume(self.local)
            self.assertEqual(list(self.local.iterdir()), [preserved])
        with patch.object(storage.os, "replace", side_effect=OSError("not supported")):
            with self.assertRaisesRegex(RuntimeError, "failed the POSIX check"):
                storage.probe_working_volume(self.local)
        self.assertEqual(list(self.local.iterdir()), [preserved])

    @unittest.skipIf(os.name == "nt", "Linux executable filesystem semantics required")
    def test_working_volume_accepts_fixed_executable_modes(self):
        chmod = Path.chmod
        for mode in (0o755, 0o777):
            def normalized_chmod(path, requested, *, follow_symlinks=True):
                return chmod(path, mode, follow_symlinks=follow_symlinks)

            with self.subTest(mode=oct(mode)), patch.object(Path, "chmod", normalized_chmod):
                storage.probe_working_volume(self.local)
            self.assertEqual(list(self.local.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "Linux executable filesystem semantics required")
    def test_working_volume_accepts_chmod_denied_when_execution_works(self):
        chmod = Path.chmod

        def fixed_chmod(path, requested, *, follow_symlinks=True):
            # Simulate an executable fixed-mode mount rejecting permission changes.
            chmod(path, 0o777, follow_symlinks=follow_symlinks)
            raise PermissionError("chmod not supported")

        with patch.object(Path, "chmod", fixed_chmod):
            storage.probe_working_volume(self.local)
        self.assertEqual(list(self.local.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "Linux executable filesystem semantics required")
    def test_working_volume_rejects_nonexecutable_files_and_cleans_up(self):
        chmod = Path.chmod
        preserved = self.write(self.local / "notebook.ipynb", b"existing user file")

        def nonexecutable_chmod(path, requested, *, follow_symlinks=True):
            return chmod(path, 0o644, follow_symlinks=follow_symlinks)

        with patch.object(Path, "chmod", nonexecutable_chmod):
            with self.assertRaisesRegex(RuntimeError, "could not execute a workspace file.*0o644"):
                storage.probe_working_volume(self.local)
        self.assertEqual(list(self.local.iterdir()), [preserved])
        self.assertEqual(preserved.read_bytes(), b"existing user file")

    @unittest.skipIf(os.name == "nt", "Linux executable filesystem semantics required")
    def test_working_volume_rejects_execution_failures_and_cleans_up(self):
        failures = (PermissionError("noexec mount"),
                    subprocess.TimeoutExpired("execution-check", 10),
                    subprocess.CalledProcessError(1, "execution-check"))
        for failure in failures:
            with self.subTest(failure=failure), patch.object(storage.subprocess, "run", side_effect=failure) as run:
                with self.assertRaisesRegex(RuntimeError, "could not execute a workspace file"):
                    storage.probe_working_volume(self.local)
                command = run.call_args.args[0]
                self.assertEqual(len(command), 1)
                self.assertTrue(Path(command[0]).is_relative_to(self.local))
                self.assertNotIn("shell", run.call_args.kwargs)
            self.assertEqual(list(self.local.iterdir()), [])

    def mounts(self, *entries):
        text = "1 0 0:1 / / rw - overlay overlay rw\n"
        for i, (path, kind) in enumerate(entries, 2):
            text += f"{i} 1 0:{i} / {Path(path).resolve().as_posix()} rw - {kind} volume rw\n"
        return patch.object(Path, "read_text", return_value=text)

    def detect(self, *entries, links=True, env=None):
        alternate = self.base / "workspace-global"
        alternate.mkdir(exist_ok=True)
        with patch.object(Path, "exists", return_value=True), self.mounts(*entries), \
                patch.object(storage, "hard_links_work", return_value=links), \
                patch.dict(os.environ, {"FLASHHEAD_STORAGE": "auto", **(env or {})}):
            return storage.detect_storage(self.local, alternate)

    def test_detect_network_volume_keeps_the_existing_layout(self):
        self.assertEqual(self.detect((self.local, "fuse.vstorage")), ("network", self.local))
        # A Network Volume at /workspace wins over a Global Volume beside it.
        alternate = self.base / "workspace-global"
        self.assertEqual(self.detect((self.local, "nfs4"), (alternate, "fuse")),
                         ("network", self.local))

    def test_detect_global_volume_by_missing_hard_links_or_mount_type(self):
        self.assertEqual(self.detect((self.local, "fuse"), links=False), ("global", self.local))
        self.assertEqual(self.detect((self.local, "fuse.global")), ("global", self.local))
        # Mounted beside an empty /workspace (its mount path edited at deploy time).
        alternate = self.base / "workspace-global"
        self.assertEqual(self.detect((alternate, "fuse")), ("global", alternate))

    def test_detect_never_takes_the_container_disk_for_a_volume(self):
        for entries in ((), ((self.local, "overlay"),), ((self.local, "tmpfs"),)):
            with self.subTest(entries=entries):
                self.assertEqual(self.detect(*entries), ("network", self.local))

    def test_detect_override(self):
        alternate = self.base / "workspace-global"
        self.assertEqual(self.detect((self.local, "fuse.vstorage"), env={"FLASHHEAD_STORAGE": "global"}),
                         ("global", self.local))
        self.assertEqual(self.detect((self.local, "nfs4"), (alternate, "fuse"), env={"FLASHHEAD_STORAGE": "global"}),
                         ("global", alternate))
        self.assertEqual(self.detect((self.local, "fuse"), links=False, env={"FLASHHEAD_STORAGE": "network"}),
                         ("network", self.local))
        with self.assertRaisesRegex(RuntimeError, "not auto, network or global"):
            self.detect((self.local, "fuse"), env={"FLASHHEAD_STORAGE": "s3"})

    @unittest.skipIf(os.name == "nt", "Linux hard links required")
    def test_hard_link_probe_cleans_up(self):
        preserved = self.write(self.local / "notebook.ipynb", b"existing user file")
        self.assertTrue(storage.hard_links_work(self.local))
        with patch.object(storage.os, "link", side_effect=PermissionError("not supported")):
            self.assertFalse(storage.hard_links_work(self.local))
        self.assertEqual(list(self.local.iterdir()), [preserved])

    def test_global_volume_needs_a_mount_and_plain_writes_only(self):
        with patch.object(Path, "exists", return_value=True), self.mounts():
            with self.assertRaisesRegex(RuntimeError, "No volume mounted"):
                storage.require_global_volume(self.local)
        with patch.object(Path, "exists", return_value=True), self.mounts((self.local, "tmpfs")):
            with self.assertRaisesRegex(RuntimeError, "not a persistent volume"):
                storage.require_global_volume(self.local)
        # No rename, link, symlink or chmod is needed on the volume.
        with patch.object(Path, "exists", return_value=True), self.mounts((self.local, "fuse")), \
                patch.object(storage.os, "replace", side_effect=OSError("no rename")), \
                patch.object(storage.os, "link", side_effect=OSError("no hard links")), \
                patch.object(storage.os, "chmod", side_effect=OSError("no permission bits")), \
                patch.object(storage.os, "symlink", side_effect=OSError("no symlinks")):
            storage.require_global_volume(self.local)
        self.assertEqual(list(self.local.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "Linux symlinks required")
    def test_link_global_data_seeds_once_and_links(self):
        comfy = self.base / "local/ComfyUI"
        volume = self.local / "runpod-slim/ComfyUI"
        self.write(comfy / "models/configs/v1-inference.yaml", b"image default")
        self.write(comfy / "models/checkpoints/put_checkpoints_here", b"")
        self.write(comfy / "input/example.png", b"image example")
        self.write(comfy / "user/comfyui.log", b"local log")
        # A file already on the volume is never replaced by the image's copy.
        own = self.write(volume / "input/example.png", b"my upload")
        storage.link_global_data(comfy, volume)
        for folder in storage.GLOBAL_DATA:
            self.assertTrue((comfy / folder).is_symlink(), folder)
            self.assertEqual((comfy / folder).resolve(), (volume / folder).resolve())
        self.assertEqual((volume / "models/configs/v1-inference.yaml").read_bytes(), b"image default")
        self.assertTrue((volume / "models/checkpoints/put_checkpoints_here").is_file())
        self.assertEqual(own.read_bytes(), b"my upload")
        # Logs and Manager caches under user/ stay on the container disk.
        self.assertFalse((comfy / "user").is_symlink())
        self.assertEqual((comfy / "user/comfyui.log").read_bytes(), b"local log")
        self.assertFalse((volume / "user/comfyui.log").exists())
        # Linked again on a later boot: nothing changes, nothing is copied.
        own.write_bytes(b"edited upload")
        storage.link_global_data(comfy, volume)
        self.assertEqual((comfy / "input/example.png").read_bytes(), b"edited upload")

    def test_workspace_paths_on_the_volume_pass(self):
        comfy = self.local / "runpod-slim/ComfyUI"
        with patch.dict(os.environ, {"HF_HOME": str(self.local / ".cache/huggingface")}):
            storage.check_workspace_paths(comfy, self.local)

    @unittest.skipIf(os.name == "nt", "Linux symlinks required")
    def test_external_data_links_fail_without_modifying_files(self):
        comfy = self.local / "ComfyUI"
        source = self.write(self.base / "previous-volume/portrait.png", b"portrait")
        comfy.mkdir()
        link = comfy / "input"
        link.symlink_to(source.parent, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "points outside"):
            storage.check_workspace_paths(comfy, self.local)
        self.assertTrue(link.is_symlink())
        self.assertEqual(source.read_bytes(), b"portrait")
        link.unlink()
        model = comfy / "models/diffusion_models/old.safetensors"
        model.parent.mkdir(parents=True)
        model.symlink_to(self.base / "missing-volume/model.safetensors")
        with self.assertRaisesRegex(RuntimeError, "points outside"):
            storage.check_workspace_paths(comfy, self.local)
        self.assertTrue(model.is_symlink())


if __name__ == "__main__":
    unittest.main(verbosity=2)
