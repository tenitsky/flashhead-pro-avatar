"""CPU checks for every bundled graph: wiring, node schemas, settings, layout, samples,
and that setup.sh downloads exactly the files the engine loads.
python tools/test_workflows.py"""
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
NODE = ROOT / "custom_nodes/comfyui-flashhead"


def load(name, path, package=None):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = "/models"
sys.modules.setdefault("folder_paths", folder_paths)
builder = load("builder", ROOT / "tools/build_workflow.py")
nodes_pack = load("comfyui_flashhead", NODE / "__init__.py", package=True)
engine = nodes_pack.engine
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")
# Widget inputs the frontend adds after an INT input named seed.
CONTROL_AFTER = {"seed"}


class WorkflowTests(unittest.TestCase):
    def each(self):
        for w in builder.WORKFLOWS:
            wf = builder.build(w["variant"], w["batch"])
            yield w, wf, {n["id"]: n for n in wf["nodes"]}

    def sources(self, wf, nodes):
        links = {}
        for lid, src, slot, dst, port, kind in wf["links"]:
            out, inp = nodes[src]["outputs"][slot], nodes[dst]["inputs"][port]
            self.assertEqual(out["type"], kind)
            self.assertEqual(inp["type"], kind)
            self.assertEqual(inp["link"], lid)
            self.assertIn(lid, out["links"])
            links[(dst, inp["name"])] = (src, out["name"])
        return links

    def test_shipped_files_match_builder(self):
        shipped = {p.name for p in (ROOT / "workflows").glob("*.json")}
        self.assertEqual(shipped, {w["path"] for w in builder.WORKFLOWS})
        self.assertEqual(len({w["id"] for w in builder.WORKFLOWS}), len(builder.WORKFLOWS))
        for w, wf, _ in self.each():
            with self.subTest(workflow=w["path"]):
                path = ROOT / "workflows" / w["path"]
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), wf)

    def test_own_nodes_match_their_schema(self):
        for w, wf, nodes in self.each():
            for node in nodes.values():
                cls = nodes_pack.NODE_CLASS_MAPPINGS.get(node["type"])
                if cls is None:
                    continue
                with self.subTest(workflow=w["path"], node=node["type"]):
                    schema = cls.INPUT_TYPES()
                    inputs = {**schema.get("required", {}), **schema.get("optional", {})}
                    self.assertEqual(tuple(o["type"] for o in node["outputs"]), cls.RETURN_TYPES)
                    self.assertEqual(tuple(o["name"] for o in node["outputs"]), cls.RETURN_NAMES)
                    sockets = [k for k, v in inputs.items()
                               if not (isinstance(v[0], list) or v[0] in
                                       ("INT", "FLOAT", "BOOLEAN", "STRING"))]
                    self.assertEqual([i["name"] for i in node["inputs"]], sockets)
                    widgets = []
                    for k, v in inputs.items():
                        if isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "BOOLEAN", "STRING"):
                            widgets.append(k)
                            if k in CONTROL_AFTER:
                                widgets.append("control_after_generate")
                    self.assertEqual(len(widgets), len(node["widgets_values"]))
                    for name, value in zip(widgets, node["widgets_values"]):
                        spec = inputs.get(name)
                        if spec and isinstance(spec[0], list):
                            self.assertIn(value, spec[0], name)

    def test_wiring(self):
        for w, wf, nodes in self.each():
            with self.subTest(workflow=w["path"]):
                links = self.sources(wf, nodes)
                expected = {(40, "portrait"): (10, "IMAGE"),
                            (40, "flashhead"): (30, "flashhead")}
                if not w["batch"]:
                    expected[(40, "audio")] = (11, "AUDIO")
                self.assertEqual(links, expected)
                self.assertEqual(nodes[40]["type"],
                                 "FlashHeadBatchFolder" if w["batch"] else "FlashHeadRender")

    def test_variant_settings_and_samples(self):
        for w, wf, nodes in self.each():
            v = builder.VARIANTS[w["variant"]]
            with self.subTest(workflow=w["path"]):
                anchor, reset_seconds, seed, control, color, suffix = \
                    nodes[40]["widgets_values"][:6]
                self.assertIn(anchor, engine.ANCHORS)
                self.assertEqual((anchor, reset_seconds, suffix),
                                 (v["anchor"], v["reset_seconds"], v["suffix"]))
                self.assertEqual((seed, control, color), (42, "fixed", 1.0))
                self.assertEqual(nodes[30]["widgets_values"], ["pro", True])
                self.assertEqual(nodes[10]["widgets_values"][0], builder.SAMPLE_PORTRAIT)
                if not w["batch"]:
                    self.assertEqual(nodes[11]["widgets_values"][0], builder.SAMPLE_VOICEOVER)
        self.assertIn(builder.ANCHOR_CONTINUOUS, engine.ANCHORS)
        # Continuous only: every chunk continues from the one before, never a reset.
        self.assertEqual([w["variant"] for w in builder.WORKFLOWS],
                         ["continuous", "continuous"])
        self.assertEqual([w["batch"] for w in builder.WORKFLOWS], [False, True])
        # The Render nodes frame the face themselves: no separate portrait node.
        for w, wf, nodes in self.each():
            self.assertNotIn("FlashHeadPortrait", {n["type"] for n in nodes.values()})

    def test_samples_ship_with_the_repo(self):
        shipped = {p.name for p in (ROOT / "samples").iterdir()}
        self.assertEqual(shipped, {builder.SAMPLE_PORTRAIT, builder.SAMPLE_VOICEOVER})
        self.assertIn('"$SCRIPT_DIR/samples/"*', SETUP)

    def test_models_match_setup_downloads(self):
        calls = re.findall(r'^download_hf "\$(\w+)" "([^"]+)" ([\w.-]+)', SETUP, re.M)
        repos = dict(re.findall(r'^(\w+_REPO)="([^"]+)"', SETUP, re.M))
        downloads = sorted((p, folder, repos[var]) for var, p, folder in calls)
        self.assertEqual(downloads, sorted(builder.MODEL_FILES))
        self.assertIn(builder.FLASHHEAD_REF, SETUP)
        self.assertIn(f"https://github.com/{builder.FLASHHEAD_REPO}.git", SETUP)
        # The folders the loader reads are the ones setup fills.
        loader = (NODE / "__init__.py").read_text(encoding="utf-8")
        for folder in {f for _, f, _ in builder.MODEL_FILES}:
            self.assertIn(f'"{folder}"', loader)
        self.assertIn(engine.YUNET.rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
                      [p.name for p in NODE.iterdir()])

    def test_layout_is_spaced_grouped_and_expanded(self):
        title = 40
        for w, wf, nodes in self.each():
            with self.subTest(workflow=w["path"]):
                boxes = {nid: (n["pos"][0], n["pos"][1] - title, n["pos"][0] + n["size"][0],
                               n["pos"][1] + n["size"][1]) for nid, n in nodes.items()}
                ids = sorted(boxes)
                for i, a in enumerate(ids):
                    self.assertNotEqual(nodes[a]["flags"].get("collapsed"), True)
                    for b in ids[i + 1:]:
                        ax0, ay0, ax1, ay1 = boxes[a]
                        bx0, by0, bx1, by1 = boxes[b]
                        self.assertFalse(ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1,
                                         (a, b))
                for group in wf["groups"]:
                    self.assertTrue(group["title"][0].isdigit())


class StandaloneNotesTests(unittest.TestCase):
    """Every workflow stands alone: its notes and titles never mention another model,
    template or workflow."""

    def test_no_cross_references(self):
        pattern = re.compile(r"\b(LongCat|Qwen|H3|MiniMax|InfiniteTalk|Lite)\b|\bworkflows? [0-9]|"
                             r"other (workflow|template)|avatar templates", re.I)
        for path in sorted((ROOT / "workflows").glob("*.json")):
            wf = json.loads(path.read_text(encoding="utf-8"))
            for node in wf["nodes"]:
                texts = [node.get("title") or ""]
                if node["type"] == "MarkdownNote":
                    texts += [v for v in node.get("widgets_values", []) if isinstance(v, str)]
                for text in texts:
                    hit = pattern.search(text)
                    self.assertIsNone(hit, f"{path.name} node {node['id']}: "
                                      f"{hit and hit.group(0)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
