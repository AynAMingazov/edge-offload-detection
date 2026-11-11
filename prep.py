# -*- coding: utf-8 -*-
"""Prepares the test data: 500 COCO val2017 images with people or cars, a simulated low-light copy,
and the JPEG variants that a vehicle would send to an edge server. Writes sizes and encode times to prep.json."""
import io, json, os, sys, time, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N = 500
SEED = 13
RES = [640, 480, 320]          # longer side of the transmitted frame
QUAL = {640: [90, 70, 50, 30, 10], 480: [70, 30], 320: [70, 30]}
cv2.setNumThreads(1)


def fetch(url, path, tries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
                f.write(r.read())
            os.replace(path + ".part", path)
            return
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 + 2 * i)


def low_light(img, rng):
    """Low-light simulation: darken in linear light by about 3 stops, add signal-dependent and read noise."""
    lin = (img.astype(np.float32) / 255.0) ** 2.2
    lin *= 0.12
    noise = rng.normal(0.0, 1.0, lin.shape).astype(np.float32)
    lin = lin + noise * np.sqrt(lin * 0.0015 + 0.00002)
    out = np.clip(lin, 0, 1) ** (1 / 2.2)
    return (out * 255.0 + 0.5).astype(np.uint8)


def resize_long(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(ROOT, exist_ok=True)
    lz = os.path.join(ROOT, "coco2017labels.zip")
    fetch("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip", lz)
    z = zipfile.ZipFile(lz)
    names = sorted(n for n in z.namelist() if "labels/val2017/" in n and n.endswith(".txt"))
    chosen = []
    for n in names:
        rows = [l.split() for l in z.read(n).decode().strip().splitlines() if l.strip()]
        if any(r[0] in ("0", "2") for r in rows):
            chosen.append((os.path.basename(n)[:-4], rows))
        if len(chosen) == N:
            break
    print("selected", len(chosen), "images")
    src = os.path.join(ROOT, "src"); os.makedirs(src, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda c: fetch("http://images.cocodataset.org/val2017/%s.jpg" % c[0], os.path.join(src, c[0] + ".jpg")), chosen))
    print("downloaded")

    def variant_dir(name):
        d = os.path.join(ROOT, name)
        os.makedirs(os.path.join(d, "images"), exist_ok=True); os.makedirs(os.path.join(d, "labels"), exist_ok=True)
        return d

    rng = np.random.default_rng(SEED)
    stats = {"n": len(chosen), "variants": {}, "objects": {"person": 0, "car": 0}}
    acc = {}
    for stem, rows in chosen:
        img = cv2.imread(os.path.join(src, stem + ".jpg"))
        label = "\n".join(" ".join(r) for r in rows) + "\n"
        stats["objects"]["person"] += sum(r[0] == "0" for r in rows)
        stats["objects"]["car"] += sum(r[0] == "2" for r in rows)
        frames = {"day": img, "night": low_light(img, rng)}
        for cond, frame in frames.items():
            d = variant_dir(cond + "_raw")
            cv2.imwrite(os.path.join(d, "images", stem + ".png"), frame)
            open(os.path.join(d, "labels", stem + ".txt"), "w").write(label)
            a = acc.setdefault(cond + "_raw", {"bytes": [], "enc": [], "dec": []})
            a["bytes"].append(frame.shape[0] * frame.shape[1] * 3)
            for res in RES:
                small = resize_long(frame, res)
                for q in QUAL[res]:
                    name = "%s_r%d_q%d" % (cond, res, q)
                    d = variant_dir(name)
                    t0 = time.perf_counter()
                    small2 = resize_long(frame, res)
                    ok, buf = cv2.imencode(".jpg", small2, [cv2.IMWRITE_JPEG_QUALITY, q])
                    t1 = time.perf_counter()
                    cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    t2 = time.perf_counter()
                    open(os.path.join(d, "images", stem + ".jpg"), "wb").write(buf.tobytes())
                    open(os.path.join(d, "labels", stem + ".txt"), "w").write(label)
                    a = acc.setdefault(name, {"bytes": [], "enc": [], "dec": []})
                    a["bytes"].append(len(buf)); a["enc"].append((t1 - t0) * 1000); a["dec"].append((t2 - t1) * 1000)
    for name, a in acc.items():
        stats["variants"][name] = {"bytes_mean": float(np.mean(a["bytes"])), "bytes_median": float(np.median(a["bytes"])),
                                   "enc_ms": float(np.median(a["enc"])) if a["enc"] else 0.0,
                                   "dec_ms": float(np.median(a["dec"])) if a["dec"] else 0.0}
    json.dump(stats, open(os.path.join(ROOT, "prep.json"), "w"), indent=1)
    print(json.dumps({k: round(v["bytes_mean"] / 1024, 1) for k, v in stats["variants"].items()}))
    print(stats["objects"])


if __name__ == "__main__":
    main()
