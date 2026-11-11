# -*- coding: utf-8 -*-
"""Accuracy runs: mAP of each model on each image variant (people and cars). Results are appended to acc.json,
so the script can be restarted."""
import json, os, sys, shutil
import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT  # noqa: F401  (import keeps ultralytics settings initialised)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results", "acc.json")
NAMES = None


def yaml_for(variant):
    global NAMES
    if NAMES is None:
        NAMES = YOLO(os.path.join(HERE, "yolo11n.pt")).names
    p = os.path.join(ROOT, variant + ".yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: %s/images\nval: %s/images\nnames:\n" % (ROOT.replace("\\", "/"), variant, variant))
        for i in range(len(NAMES)):
            f.write("  %d: %s\n" % (i, NAMES[i]))
    return p


def jobs():
    j = []
    for cond in ("day", "night"):
        for m in ("yolo11n", "yolov8n", "yolo11s"):          # on-board candidates, raw frame
            for size in (640, 480, 320):
                if m != "yolo11n" and size != 640:
                    continue
                j.append((m, cond + "_raw", size))
        for m in ("yolo11s", "yolo11m"):                      # edge server, compressed frame
            for res, quals in ((640, (90, 70, 50, 30, 10)), (480, (70, 30)), (320, (70, 30))):
                for q in quals:
                    j.append((m, "%s_r%d_q%d" % (cond, res, q), res))
        j.append(("yolo11m", cond + "_raw", 640))             # upper bound: large model, uncompressed frame
    return j


def main():
    torch.set_num_threads(8)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for m, variant, size in jobs():
        key = "%s|%s|%d" % (m, variant, size)
        if key in done:
            continue
        model = YOLO(os.path.join(HERE, m + ".pt"))
        r = model.val(data=yaml_for(variant), imgsz=size, batch=1, device="cpu", classes=[0, 2], conf=0.001, iou=0.7,
                      plots=False, verbose=False, workers=0, project=os.path.join(HERE, "runs"), name="v", exist_ok=True)
        per = {int(c): float(v) for c, v in zip(r.box.ap_class_index, r.box.ap.tolist() if hasattr(r.box.ap, "tolist") else r.box.ap)}
        per50 = {int(c): float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50.tolist())}
        done[key] = {"map": float(r.box.map), "map50": float(r.box.map50), "recall": float(r.box.mr), "precision": float(r.box.mp),
                     "ap_person": per.get(0), "ap_car": per.get(2), "ap50_person": per50.get(0), "ap50_car": per50.get(2),
                     "val_inf_ms": float(r.speed["inference"])}
        json.dump(done, open(OUT, "w"), indent=1)
        print(key, "mAP50-95 %.3f mAP50 %.3f" % (done[key]["map"], done[key]["map50"]), flush=True)
    shutil.rmtree(os.path.join(HERE, "runs"), ignore_errors=True)
    print("ALL DONE", len(done))


if __name__ == "__main__":
    main()
