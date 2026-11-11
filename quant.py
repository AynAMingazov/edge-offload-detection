# -*- coding: utf-8 -*-
"""INT8 post-training quantisation (OpenVINO + NNCF) of the on-board and roadside models.
Calibration uses 300 other COCO val2017 images (not in the test set), half of them in simulated low light.
Steps: python quant.py calib | export | acc | time <threads>"""
import json, os, sys, glob, time, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data")
MODELS = ("yolo11n", "yolo11s")


def calib():
    import numpy as np, cv2
    sys.path.insert(0, HERE)
    from prep import fetch, low_light
    from concurrent.futures import ThreadPoolExecutor
    test = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "day_raw", "labels", "*.txt"))}
    z = zipfile.ZipFile(os.path.join(ROOT, "coco2017labels.zip"))
    names = sorted(n for n in z.namelist() if "labels/val2017/" in n and n.endswith(".txt"))
    chosen = []
    for n in names:
        stem = os.path.basename(n)[:-4]
        if stem in test:
            continue
        rows = [l.split() for l in z.read(n).decode().strip().splitlines() if l.strip()]
        if any(r[0] in ("0", "2") for r in rows):
            chosen.append((stem, rows))
        if len(chosen) == 300:
            break
    src = os.path.join(ROOT, "calib_src"); os.makedirs(src, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda c: fetch("http://images.cocodataset.org/val2017/%s.jpg" % c[0], os.path.join(src, c[0] + ".jpg")), chosen))
    d = os.path.join(ROOT, "calib")
    for s in ("images", "labels"):
        os.makedirs(os.path.join(d, s), exist_ok=True)
    rng = np.random.default_rng(29)
    for i, (stem, rows) in enumerate(chosen):
        img = cv2.imread(os.path.join(src, stem + ".jpg"))
        if i % 2:
            img = low_light(img, rng)
        cv2.imwrite(os.path.join(d, "images", stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(os.path.join(d, "labels", stem + ".txt"), "w").write("\n".join(" ".join(r) for r in rows) + "\n")
    print("CALIB", len(chosen), "overlap with test:", len(test & {c[0] for c in chosen}))


def yaml(variant):
    from ultralytics import YOLO
    names = YOLO(os.path.join(HERE, "yolo11n.pt")).names
    p = os.path.join(ROOT, variant + ".yaml")
    with open(p, "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: %s/images\nval: %s/images\nnames:\n" % (ROOT.replace("\\", "/"), variant, variant))
        for i in range(len(names)):
            f.write("  %d: %s\n" % (i, names[i]))
    return p


def export():
    from ultralytics import YOLO
    import shutil
    for m in MODELS:
        for kind in ("fp32", "int8"):
            out = os.path.join(HERE, "ov", "%s_%s_openvino_model" % (m, kind))
            if os.path.exists(os.path.join(out, m + ".xml")):
                continue
            kw = dict(format="openvino", imgsz=640, dynamic=False, half=False)
            if kind == "int8":
                kw.update(int8=True, data=yaml("calib"), fraction=1.0)
            p = YOLO(os.path.join(HERE, m + ".pt")).export(**kw)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.rmtree(out, ignore_errors=True)
            shutil.move(p, out)
            meta = os.path.join(out, "metadata.yaml")
            for f in os.listdir(out):
                if f.endswith(".xml"):
                    os.replace(os.path.join(out, f), os.path.join(out, m + ".xml"))
                if f.endswith(".bin"):
                    os.replace(os.path.join(out, f), os.path.join(out, m + ".bin"))
            print("EXPORTED", m, kind, os.path.exists(meta), flush=True)


def acc():
    from ultralytics import YOLO
    OUT = os.path.join(HERE, "results", "acc_q.json")
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs = []
    for cond in ("day", "night"):
        for kind in ("fp32", "int8"):
            jobs.append(("yolo11n", kind, cond + "_raw"))
            jobs.append(("yolo11s", kind, cond + "_r640_q70"))
    for m, kind, v in jobs:
        key = "%s_%s|%s|640" % (m, kind, v)
        if key in done:
            continue
        r = YOLO(os.path.join(HERE, "ov", "%s_%s_openvino_model" % (m, kind)), task="detect").val(
            data=yaml(v), imgsz=640, batch=1, device="cpu", classes=[0, 2], conf=0.001, iou=0.7, plots=False,
            verbose=False, workers=0, project=os.path.join(HERE, "runs"), name="q", exist_ok=True)
        done[key] = {"map": float(r.box.map), "map50": float(r.box.map50), "recall": float(r.box.mr)}
        json.dump(done, open(OUT, "w"), indent=1)
        print(key, "%.3f" % done[key]["map"], flush=True)
    print("QACC DONE", len(done))


def timing(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import numpy as np, cv2, openvino as ov
    orig = ov.Core.compile_model

    def patched(self, model, device_name=None, config=None, **kw):     # pin OpenVINO to the given number of threads
        config = dict(config or {}); config["INFERENCE_NUM_THREADS"] = threads; config["PERFORMANCE_HINT"] = "LATENCY"
        return orig(self, model, device_name or "CPU", config, **kw)
    ov.Core.compile_model = patched
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    OUT = os.path.join(HERE, "results", "time_q.json")
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    frames = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(ROOT, "day_raw", "images", "*.png")))[:110]]
    for m in MODELS:
        for kind in ("fp32", "int8"):
            key = "%s_%s|640|%d" % (m, kind, threads)
            if key in done:
                continue
            model = YOLO(os.path.join(HERE, "ov", "%s_%s_openvino_model" % (m, kind)), task="detect")
            for f in frames[:10]:
                model.predict(f, imgsz=640, device="cpu", verbose=False, classes=[0, 2])
            ts = []
            for f in frames[10:110]:
                t0 = time.perf_counter()
                model.predict(f, imgsz=640, device="cpu", verbose=False, classes=[0, 2])
                ts.append((time.perf_counter() - t0) * 1000)
            done[key] = {"total_ms_median": float(np.median(ts)), "total_ms_p90": float(np.percentile(ts, 90))}
            json.dump(done, open(OUT, "w"), indent=1)
            print(key, "%.1f ms" % done[key]["total_ms_median"], flush=True)
    print("QTIME DONE", threads)


if __name__ == "__main__":
    {"calib": calib, "export": export, "acc": acc}.get(sys.argv[1], lambda: timing(int(sys.argv[2])))()
