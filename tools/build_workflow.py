#!/usr/bin/env python3
"""
Generate the FlashHead Pro workflow JSON files. Re-run after editing:

    python tools/build_workflow.py

Every workflow is the same small graph: Load Portrait -> FlashHead Portrait (face crop)
-> FlashHead Render, with the model loader and the voiceover. They differ only in the
anchor (how chunks join): continuous, or continuous with a refresh from the portrait at
a real pause every 20-40 s; the batch version renders a folder of voiceovers with refreshes.
Each opens on the shipped samples.

Layout: numbered stages from left to right, each a coloured group with a note
underneath, and a "Start here" note on the far left. Nothing is collapsed.
"""

import json
import os

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "workflows")

FLASHHEAD_REPO = "Soul-AILab/SoulX-FlashHead"
FLASHHEAD_REF = "9bc03de06bb0de82cd6bc477804512ae06144bf2"
LF = "comfyui-flashhead"

# (path inside the Hugging Face repo, ComfyUI/models/flashhead/<folder>, repo)
MODEL_FILES = [
    ("Model_Pro/config.json", "SoulX-FlashHead-1_3B", "Soul-AILab/SoulX-FlashHead-1_3B"),
    ("Model_Pro/diffusion_pytorch_model.safetensors", "SoulX-FlashHead-1_3B",
     "Soul-AILab/SoulX-FlashHead-1_3B"),
    ("VAE_Wan/Wan2.1_VAE.pth", "SoulX-FlashHead-1_3B", "Soul-AILab/SoulX-FlashHead-1_3B"),
    ("config.json", "wav2vec2-base-960h", "facebook/wav2vec2-base-960h"),
    ("preprocessor_config.json", "wav2vec2-base-960h", "facebook/wav2vec2-base-960h"),
    ("model.safetensors", "wav2vec2-base-960h", "facebook/wav2vec2-base-960h"),
]

SAMPLE_PORTRAIT = "sample_portrait.png"
SAMPLE_VOICEOVER = "sample_voiceover.mp3"

ANCHOR_CONTINUOUS = "continuous (upstream)"
ANCHOR_PAUSES = "refresh from portrait at a pause"
FRAMING_FACE = "face crop"

VARIANTS = {
    "continuous": dict(
        title="Continuous", anchor=ANCHOR_CONTINUOUS, reset_seconds=20.0,
        suffix="_pro_continuous",
        summary="Every 1.12 s chunk continues from the previous one's last 5 frames, as "
                "upstream does: seamless, with no pose resets. Over the 71 s sample it "
                "did not drift."),
    "refresh_at_pauses": dict(
        title="Refresh at Pauses (20-40 s)", anchor=ANCHOR_PAUSES, reset_seconds=20.0,
        suffix="_pro_refresh",
        summary="Continuous, plus a full refresh against drift on long videos: once in "
                "every 20-40 s, in the middle of a real pause in the speech (mouth "
                "shut), the render starts again from the portrait. The cuts are planned "
                "from the whole voiceover before rendering. A small jump in a silence, in "
                "exchange for a face that cannot wander over minutes."),
}

WORKFLOWS = [
    dict(variant="continuous", batch=False, path="flashhead_pro_1_continuous.json",
         id="b6884540-f6dc-46dd-9c9f-b8185325d314"),
    dict(variant="refresh_at_pauses", batch=False,
         path="flashhead_pro_2_refresh_at_pauses.json",
         id="db345799-55fc-4d16-99c4-34a0f27c2d09"),
    dict(variant="refresh_at_pauses", batch=True,
         path="flashhead_pro_3_batch_folder_refresh_at_pauses.json",
         id="598d1f1f-eea6-40bf-bfa8-ea15b169aea4"),
]

# ------------------------------------------------------------------------ notes
NOTE_ABOUT = """## About SoulX-FlashHead

Soul-AILab's 1.3B audio-driven talking-head model (Apache-2.0, including the Wan 2.1 VAE). **Pro** is the higher-quality variant: the same 512 x 512 at 25 fps as Lite, but decoded with the Wan 2.1 VAE instead of Lite's much more compressed LTX VAE. Upstream: 10.8 FPS on one RTX 4090, about 2.3 s per second of video.

It generates **33-frame chunks**. Each chunk opens on the previous chunk's last **5 frames** and adds **28 new ones** (1.12 s), with the portrait as identity reference throughout. That is video continuation, like LongCat, not H3's independent chunks.

One run renders the whole voiceover; there is nothing to queue chunk by chunk.

Code: https://github.com/Soul-AILab/SoulX-FlashHead"""

NOTE_PORTRAIT = """## 2. Portrait

FlashHead was trained on **512 x 512 face crops**. **face crop** frames a square twice the face width around the face (upstream's framing, found with OpenCV). **whole image** center-crops the picture instead, keeping shoulders and background but making the face small.

The preview shows exactly what the model sees. `face_ratio` widens the face crop (2.0 = upstream)."""

NOTE_MODEL = """## 3. Model - loaded once

Pro model (6.0 GB), the Wan 2.1 VAE and the wav2vec2 audio encoder, downloaded by setup to `ComfyUI/models/flashhead/`. It stays loaded between runs.

**compile** (torch.compile) makes every chunk faster after a one-off warm-up of a minute or two on the first run. If it errors, turn it off."""


def note_render(v, batch):
    where = ("Each file in `ComfyUI/input/batch_audio/` becomes `output/<file name>"
             if batch else "The video is saved as `output/<audio name>")
    return f"""## 4. Render

**anchor: {v['anchor']}**. {v['summary']}

{where}{v['suffix']}.mp4`, with your original audio. Timing (seconds of rendering per second of video, peak VRAM) is printed in the log.

**Watch it as it renders.** The Render node shows the newest frame live, and the video is saved in ~30 s parts you can play in JupyterLab while the rest renders: `output/flashhead_parts/<name>/part_001.mp4`, ... **Cancel** to stop; queue again with the same inputs and it resumes after the last finished part. The parts are removed when the video is done.

**color_correction** pulls each chunk's colours toward the portrait (upstream 1.0). **seed** is fixed, so two renders can be compared on identical noise."""


def note_start(w, v):
    if w["batch"]:
        audio = ("2. Put voiceovers in `ComfyUI/input/batch_audio/` (Jupyter, port 8888). "
                 "Files already rendered are skipped.")
    else:
        audio = ("2. **Load Voiceover** - opens on the 71 s sample; any length works.")
    return f"""# FlashHead Pro - {v['title']}{' - Batch Folder' if w['batch'] else ''}

{v['summary']}

**Red nodes are the ones you set.** Stages run left to right with a note under each.

1. **Load Portrait** - opens on the sample portrait.
{audio}
3. Check the **portrait preview** (stage 2): face crop or whole image.
4. **Run.** One run renders the whole video and shows it on the Render node.

Long voiceovers (minutes): use workflow 2, which refreshes from the portrait in the middle of a real pause every 20-40 s.

Full instructions: https://github.com/tenitsky/flashhead-pro-avatar"""


# ------------------------------------------------------------------------ graph
class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._link_id = 0

    def node(self, nid, ntype, inputs=(), outputs=(), widgets=None, title=None,
             color=None, cnr="comfy-core", mode=0, size=(440, 100)):
        props = {"Node name for S&R": ntype}
        if cnr:
            props["cnr_id"] = cnr
        n = {
            "id": nid, "type": ntype, "pos": [0, 0], "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": mode,
            "inputs": [dict({"localized_name": nm, "name": nm, "type": tp, "link": None},
                            **({"widget": {"name": nm}} if wid else {}),
                            **({"shape": 7} if opt else {}))
                       for nm, tp, wid, opt in inputs],
            "outputs": [{"localized_name": nm, "name": nm, "type": tp, "links": []}
                        for nm, tp in outputs],
            "properties": props,
        }
        if title:
            n["title"] = title
        if widgets is not None:
            n["widgets_values"] = widgets
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return nid

    def link(self, src, src_slot, dst, dst_name):
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        if isinstance(src_slot, str):
            src_slot = next(i for i, o in enumerate(s["outputs"]) if o["name"] == src_slot)
        out = s["outputs"][src_slot]
        slot = next(i for i, inp in enumerate(d["inputs"]) if inp["name"] == dst_name)
        inp = d["inputs"][slot]
        if inp["link"] is not None:
            raise ValueError(f"{d['type']}.{dst_name} already linked")
        if inp["type"] != out["type"]:
            raise ValueError(f"type mismatch {s['type']}.{out['name']} ({out['type']})"
                             f" -> {d['type']}.{dst_name} ({inp['type']})")
        self._link_id += 1
        lid = self._link_id
        inp["link"] = lid
        out["links"].append(lid)
        self.links.append([lid, src, src_slot, dst, slot, out["type"]])


def md(g, nid, title, text, size):
    g.node(nid, "MarkdownNote", widgets=[text], title=title, cnr=None,
           color=("#222", "#000"), size=size)


COLUMN_W, NODE_W, GAP_X, GAP_Y, TITLE_H, PAD = 520, 460, 140, 50, 120, 30
STAGE_COLORS = {"input": "#a1309b", "portrait": "#b06634", "models": "#3f789e",
                "output": "#8a4a4a"}


def lay_out(g, stages, start_notes):
    """Place stages as spaced columns, each a titled group with notes under it."""
    nodes = {n["id"]: n for n in g.nodes}
    groups = []
    x = 0
    y = 0
    for nid in start_notes:
        nodes[nid]["pos"] = [x, y]
        y += nodes[nid]["size"][1] + GAP_Y
    x += COLUMN_W + GAP_X
    for gid, (title, color, ids, notes) in enumerate(stages, 1):
        y = TITLE_H
        for nid in ids:
            nodes[nid]["pos"] = [x + PAD, y]
            y += nodes[nid]["size"][1] + GAP_Y
        bottom = y - GAP_Y + PAD
        groups.append({"id": gid, "title": title,
                       "bounding": [x, 0, COLUMN_W, bottom],
                       "color": STAGE_COLORS[color], "font_size": 26, "flags": {}})
        y = bottom + 60
        for nid in notes:
            nodes[nid]["pos"] = [x, y]
            nodes[nid]["size"][0] = COLUMN_W
            y += nodes[nid]["size"][1] + GAP_Y
        x += COLUMN_W + GAP_X
    return groups


def build(variant="continuous", batch=False):
    v = VARIANTS[variant]
    w = next(w for w in WORKFLOWS if w["variant"] == variant and w["batch"] == batch)
    g = Graph()
    INPUT = ("#322", "#533")    # red: things you set
    I, W = False, True          # socket input / widget converted to input

    def size(h):
        return (NODE_W, h)

    md(g, 1, "Start here", note_start(w, v), (COLUMN_W, 720))
    md(g, 2, "About FlashHead", NOTE_ABOUT, (COLUMN_W, 560))
    md(g, 4, "About stage 2", NOTE_PORTRAIT, (COLUMN_W, 380))
    md(g, 5, "About stage 3", NOTE_MODEL, (COLUMN_W, 320))
    md(g, 6, "About stage 4", note_render(v, batch), (COLUMN_W, 440))

    # 1. inputs
    g.node(10, "LoadImage", outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
           widgets=[SAMPLE_PORTRAIT, "image"], title="Load Portrait", color=INPUT,
           size=size(500))
    if not batch:
        g.node(11, "LoadAudio", outputs=[("AUDIO", "AUDIO")],
               widgets=[SAMPLE_VOICEOVER, None, None], title="Load Voiceover",
               color=INPUT, size=size(150))

    # 2. portrait
    g.node(20, "FlashHeadPortrait", inputs=[("image", "IMAGE", I, False)],
           outputs=[("portrait", "IMAGE")], widgets=[FRAMING_FACE, 2.0],
           title="Portrait framing (face crop or whole image)", color=INPUT, cnr=LF,
           size=size(110))
    g.node(21, "PreviewImage", inputs=[("images", "IMAGE", I, False)],
           title="What FlashHead sees (512 x 512)", size=size(500))

    # 3. model
    g.node(30, "FlashHeadLoader", outputs=[("flashhead", "FLASHHEAD")],
           widgets=["pro", True], title="FlashHead Pro (loaded once)", cnr=LF,
           size=size(110))

    # 4. render
    common = [v["anchor"], v["reset_seconds"], 42, "fixed", 1.0, v["suffix"]]
    if batch:
        g.node(40, "FlashHeadBatchFolder",
               inputs=[("flashhead", "FLASHHEAD", I, False),
                       ("portrait", "IMAGE", I, False)],
               outputs=[("summary", "STRING")], widgets=common + ["batch_audio"],
               title="Render every voiceover in input/batch_audio", color=INPUT, cnr=LF,
               size=size(560))
    else:
        g.node(40, "FlashHeadRender",
               inputs=[("flashhead", "FLASHHEAD", I, False), ("portrait", "IMAGE", I, False),
                       ("audio", "AUDIO", I, False)],
               outputs=[("video_path", "STRING")], widgets=common,
               title="Render the whole video", cnr=LF, size=size(560))

    L = g.link
    L(10, 0, 20, "image")
    L(20, 0, 21, "images")
    L(20, 0, 40, "portrait")
    L(30, 0, 40, "flashhead")
    if not batch:
        L(11, 0, 40, "audio")

    stages = [
        ("1. Your inputs", "input", [10] + ([] if batch else [11]), []),
        ("2. Portrait", "portrait", [20, 21], [4]),
        ("3. Model", "models", [30], [5]),
        ("4. Render", "output", [40], [6]),
    ]
    groups = lay_out(g, stages, [1, 2])
    return {
        "id": w["id"],
        "revision": 0,
        "last_node_id": max(n["id"] for n in g.nodes),
        "last_link_id": g._link_id,
        "nodes": g.nodes,
        "links": g.links,
        "groups": groups,
        "config": {},
        "extra": {"ds": {"scale": 0.6, "offset": [80, 80]}},
        "version": 0.4,
    }


if __name__ == "__main__":
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    for w in WORKFLOWS:
        wf = build(w["variant"], w["batch"])
        path = os.path.join(WORKFLOW_DIR, w["path"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {os.path.normpath(path)}: {len(wf['nodes'])} nodes, "
              f"{len(wf['links'])} links")
