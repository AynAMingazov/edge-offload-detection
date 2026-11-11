# -*- coding: utf-8 -*-
"""Validation on real low-light photographs (ExDark, Loh and Chan 2019).
Test split images that contain people or cars; ExDark People -> COCO person (0), Car -> COCO car (2).
The frame is first brought to a longer side of 640 px (as in the COCO set), then the same JPEG variants are built.
Usage: python exdark.py prep <img_dir> <gt_dir> <imageclasslist.txt> | acc"""
import json, os, sys, glob, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data_exdark")
QUAL = {640: [90, 70, 50, 30, 10], 480: [70, 30], 320: [70, 30]}
MAP = {"People": 0, "Car": 2}


def prep(img_dir, gt_dir, lst):
    import numpy as np, cv2
    sys.path.insert(0, HERE)
    from prep import resize_long
    cv2.setNumThreads(1)
    split = {}
    for line in open(lst, encoding="utf-8", errors="ignore"):
        p = line.split()
        if len(p) >= 5 and p[0].lower() != "name":
            split[p[0].lower()] = (int(p[4]), int(p[3]), int(p[2]))       # set, indoor/outdoor, lighting
    imgs = {os.path.basename(f).lower(): f for f in glob.glob(os.path.join(img_dir, "**", "*"), recursive=True)
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")}
    gts = {}
    for f in glob.glob(os.path.join(gt_dir, "**", "*.txt"), recursive=True):
        gts.setdefault(os.path.basename(f)[:-4].lower(), []).append(f)
    chosen = []
    for name in sorted(imgs):
        if split.get(name, (0,))[0] != 3 or name not in gts:
            continue
        boxes = set()
        for g in gts[name]:
            for l in open(g, encoding="utf-8", errors="ignore").read().splitlines()[1:]:
                p = l.split()
                if len(p) >= 5 and p[0] in MAP:
                    boxes.add((MAP[p[0]],) + tuple(float(x) for x in p[1:5]))
        if boxes:
            chosen.append((name, sorted(boxes)))
    print("test images with people/cars:", len(chosen), flush=True)
    stats = {"n": 0, "objects": {"person": 0, "car": 0}, "variants": {}, "outdoor": 0}
    accs = {}
    for name, boxes in chosen:
        img = cv2.imread(imgs[name])
        if img is None:
            continue
        H, W = img.shape[:2]
        rows = []
        for c, l, t, w, h in boxes:
            l, t = max(0.0, l), max(0.0, t); w, h = min(w, W - l), min(h, H - t)
            if w > 1 and h > 1:
                rows.append("%d %.6f %.6f %.6f %.6f" % (c, (l + w / 2) / W, (t + h / 2) / H, w / W, h / H))
        if not rows:
            continue
        stem = os.path.splitext(name)[0]
        stats["n"] += 1; stats["outdoor"] += split[name][1] == 2
        stats["objects"]["person"] += sum(r.startswith("0 ") for r in rows)
        stats["objects"]["car"] += sum(r.startswith("2 ") for r in rows)
        frame = resize_long(img, 640)
        label = "\n".join(rows) + "\n"

        def put(v, data, ext):
            d = os.path.join(ROOT, v)
            for s in ("images", "labels"):
                os.makedirs(os.path.join(d, s), exist_ok=True)
            open(os.path.join(d, "images", stem + ext), "wb").write(data)
            open(os.path.join(d, "labels", stem + ".txt"), "w").write(label)
        ok, png = cv2.imencode(".png", frame); put("raw", png.tobytes(), ".png")
        for res, qs in QUAL.items():
            for q in qs:
                small = resize_long(frame, res)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
                v = "r%d_q%d" % (res, q); put(v, buf.tobytes(), ".jpg")
                accs.setdefault(v, []).append(len(buf))
    stats["variants"] = {v: {"bytes_mean": float(np.mean(b))} for v, b in accs.items()}
    json.dump(stats, open(os.path.join(ROOT, "prep.json"), "w"), indent=1)
    print(json.dumps(stats["objects"]), stats["n"], "images,", stats["outdoor"], "outdoor")
    print({v: round(s["bytes_mean"] / 1024, 1) for v, s in stats["variants"].items()})


def acc():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(8)
    names = YOLO(os.path.join(HERE, "yolo11n.pt")).names
    OUT = os.path.join(HERE, "results", "acc_exdark.json")
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs = [("yolo11n", "raw", 640), ("yolo11n", "raw", 480), ("yolo11n", "raw", 320), ("yolov8n", "raw", 640),
            ("yolo11s", "raw", 640), ("yolo11m", "raw", 640)]
    for m in ("yolo11s", "yolo11m"):
        for res, qs in QUAL.items():
            for q in qs:
                jobs.append((m, "r%d_q%d" % (res, q), res))
    for m, v, size in jobs:
        key = "%s|%s|%d" % (m, v, size)
        if key in done:
            continue
        y = os.path.join(ROOT, v + ".yaml")
        with open(y, "w", encoding="utf-8") as f:
            f.write("path: %s\ntrain: %s/images\nval: %s/images\nnames:\n" % (ROOT.replace("\\", "/"), v, v))
            for i in range(len(names)):
                f.write("  %d: %s\n" % (i, names[i]))
        r = YOLO(os.path.join(HERE, m + ".pt")).val(data=y, imgsz=size, batch=1, device="cpu", classes=[0, 2], conf=0.001,
                                                     iou=0.7, plots=False, verbose=False, workers=0,
                                                     project=os.path.join(HERE, "runs"), name="x", exist_ok=True)
        per = {int(c): float(a) for c, a in zip(r.box.ap_class_index, r.box.ap.tolist())}
        done[key] = {"map": float(r.box.map), "map50": float(r.box.map50), "ap_person": per.get(0), "ap_car": per.get(2)}
        json.dump(done, open(OUT, "w"), indent=1)
        print(key, "%.3f" % done[key]["map"], flush=True)
    print("EXDARK DONE", len(done))


if __name__ == "__main__":
    prep(*sys.argv[2:5]) if sys.argv[1] == "prep" else acc()
