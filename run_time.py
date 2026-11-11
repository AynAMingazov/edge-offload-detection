# -*- coding: utf-8 -*-
"""Latency runs. One process per thread count: python run_time.py <threads>.
Measures median inference time per frame (batch 1, CPU) for every model and input size."""
import json, os, sys, time, glob
threads = int(sys.argv[1])
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
import numpy as np
import cv2
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "time.json")
torch.set_num_threads(threads)
cv2.setNumThreads(1)
files = sorted(glob.glob(os.path.join(HERE, "data", "day_raw", "images", "*.png")))[:120]
frames = [cv2.imread(f) for f in files]
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
plan = [("yolo11n", 640), ("yolo11n", 480), ("yolo11n", 320), ("yolov8n", 640), ("yolo11s", 640), ("yolo11s", 480),
        ("yolo11s", 320), ("yolo11m", 640), ("yolo11m", 480), ("yolo11m", 320)]
for m, size in plan:
    key = "%s|%d|%d" % (m, size, threads)
    if key in done:
        continue
    model = YOLO(os.path.join(HERE, m + ".pt"))
    for f in frames[:10]:                                    # warm-up
        model.predict(f, imgsz=size, device="cpu", verbose=False, classes=[0, 2])
    torch.set_num_threads(threads)
    ts, inf = [], []
    for f in frames[10:110]:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=size, device="cpu", verbose=False, classes=[0, 2])
        ts.append((time.perf_counter() - t0) * 1000)
        inf.append(r[0].speed["inference"])
    done[key] = {"total_ms_median": float(np.median(ts)), "total_ms_p90": float(np.percentile(ts, 90)),
                 "inference_ms_median": float(np.median(inf)), "threads_seen": torch.get_num_threads()}
    json.dump(done, open(OUT, "w"), indent=1)
    print(key, "%.1f ms (p90 %.1f), net only %.1f" % (done[key]["total_ms_median"], done[key]["total_ms_p90"], done[key]["inference_ms_median"]), flush=True)
print("TIME DONE", threads)
