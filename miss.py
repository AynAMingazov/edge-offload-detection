# -*- coding: utf-8 -*-
"""Safety-oriented metric: share of missed pedestrians (and cars) at a working confidence threshold of 0.25,
IoU >= 0.5, greedy matching. Objects are split by box height in the original frame:
small < 64 px (far), large >= 64 px (near). Usage: python miss.py"""
import json, os, glob
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "miss.json")
CONF, IOU, SMALL = 0.25, 0.5, 64
SETS = {"coco_day": ("data", "day_"), "coco_night": ("data", "night_"), "exdark": ("data_exdark", "")}
CFG = [  # name, weights, variant suffix, imgsz
    ("board_n_fp32", "yolo11n.pt", "raw", 640),
    ("board_n_int8", "ov/yolo11n_int8_openvino_model", "raw", 640),
    ("board_s_int8", "ov/yolo11s_int8_openvino_model", "raw", 640),
    ("road_s_q70", "yolo11s.pt", "r640_q70", 640),
    ("road_s_q90", "yolo11s.pt", "r640_q90", 640),
    ("road_m_q90", "yolo11m.pt", "r640_q90", 640),
]


def iou(b, g):
    x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
    x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); c = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (a[:, None] + c[None, :] - inter + 1e-9)


def run(model, root, variant, size, raw_dir):
    imgs = sorted(glob.glob(os.path.join(HERE, root, variant, "images", "*")))
    st = {c: {"small": [0, 0], "large": [0, 0]} for c in (0, 2)}   # [found, total]
    fp = {0: 0, 2: 0}
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(HERE, root, variant, "labels", stem + ".txt")
        rows = [l.split() for l in open(lab).read().splitlines() if l.strip()]
        rows = [r for r in rows if r[0] in ("0", "2")]
        # original size from the uncompressed frame, so object size does not depend on the transmitted resolution
        import cv2
        H, W = cv2.imread(os.path.join(raw_dir, stem + ".png")).shape[:2] if raw_dir else (None, None)
        r = model.predict(p, imgsz=size, conf=CONF, device="cpu", verbose=False, classes=[0, 2])[0]
        h, w = r.orig_shape
        boxes = r.boxes.xyxyn.numpy() if len(r.boxes) else np.zeros((0, 4)); cls = r.boxes.cls.numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
        for c in (0, 2):
            g = np.array([[float(x) for x in rr[1:5]] for rr in rows if int(rr[0]) == c]).reshape(-1, 4)
            gx = np.stack([g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2, g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], 1) if len(g) else np.zeros((0, 4))
            b = boxes[cls == c]
            found = np.zeros(len(gx), bool); used = np.zeros(len(b), bool)
            if len(gx) and len(b):
                m = iou(b, gx)
                for i, j in sorted(((i, j) for i in range(len(b)) for j in range(len(gx)) if m[i, j] >= IOU), key=lambda t: -m[t]):
                    if not used[i] and not found[j]:
                        used[i] = found[j] = True
            fp[c] += int((~used).sum())
            for j in range(len(gx)):
                k = "small" if g[j, 3] * H < SMALL else "large"
                st[c][k][1] += 1; st[c][k][0] += int(found[j])
    return {"person": st[0], "car": st[2], "fp_person": fp[0], "fp_car": fp[2], "images": len(imgs)}


def main():
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for sname, (root, pre) in SETS.items():
        raw_dir = os.path.join(HERE, root, pre + "raw", "images")
        for name, wts, var, size in CFG:
            key = sname + "|" + name
            if key in done:
                continue
            model = YOLO(os.path.join(HERE, wts), task="detect")
            done[key] = run(model, root, pre + var, size, raw_dir)
            json.dump(done, open(OUT, "w"), indent=1)
            p = done[key]["person"]
            miss = 1 - (p["small"][0] + p["large"][0]) / max(1, p["small"][1] + p["large"][1])
            print(key, "missed pedestrians %.1f%%" % (100 * miss), "small", p["small"], "large", p["large"], flush=True)
    print("MISS DONE", len(done))


if __name__ == "__main__":
    main()
