# -*- coding: utf-8 -*-
"""Builds tables, the end-to-end latency model and figures from acc.json, time.json and data/prep.json."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "results", "acc.json")))
T = json.load(open(os.path.join(HERE, "results", "time.json")))
P = json.load(open(os.path.join(HERE, "data", "prep.json") if os.path.exists(os.path.join(HERE, "data", "prep.json")) else os.path.join(HERE, "results", "frames.json")))
FIG = os.path.join(HERE, "fig"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})

out = {"models": {}}
for m in ("yolo11n", "yolov8n", "yolo11s", "yolo11m"):
    from ultralytics.utils.torch_utils import get_flops
    net = YOLO(os.path.join(HERE, m + ".pt")).model
    net.fuse() if hasattr(net, "fuse") else None
    out["models"][m] = {"params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2), "gflops": round(get_flops(net, 640), 1)}
print(out["models"])

def acc(m, v, s): return A["%s|%s|%d" % (m, v, s)]["map"]
def tm(m, s, th): return T["%s|%d|%d" % (m, s, th)]["total_ms_median"]
def kb(v): return P["variants"][v]["bytes_mean"] / 1024

# end-to-end latency of the offloaded variant: encode + uplink + decode + server inference
def e2e(m, cond, res, q, mbps, server_threads=8):
    v = "%s_r%d_q%d" % (cond, res, q)
    pv = P["variants"][v]
    tx = pv["bytes_mean"] * 8 / (mbps * 1e6) * 1000
    return pv["enc_ms"] + tx + pv["dec_ms"] + tm(m, res, server_threads)

rows = []
for cond in ("day", "night"):
    for th in (1, 2, 4):
        rows.append(("onboard", "yolo11n", cond, 640, None, th, acc("yolo11n", cond + "_raw", 640), tm("yolo11n", 640, th), 0.0))
        rows.append(("onboard", "yolo11n", cond, 480, None, th, acc("yolo11n", cond + "_raw", 480), tm("yolo11n", 480, th), 0.0))
    for m in ("yolo11s", "yolo11m"):
        for res, q in ((640, 70), (640, 50), (480, 70), (320, 70)):
            for mbps in (5, 20, 50):
                v = "%s_r%d_q%d" % (cond, res, q)
                rows.append(("edge", m, cond, res, q, mbps, acc(m, v, res), e2e(m, cond, res, q, mbps), kb(v)))
out["e2e"] = rows

# break-even uplink for edge yolo11s 640 q70 against on-board yolo11n 640 with 1 and 2 threads
be = {}
for cond in ("day", "night"):
    pv = P["variants"]["%s_r640_q70" % cond]
    fixed = pv["enc_ms"] + pv["dec_ms"] + tm("yolo11s", 640, 8)
    for th in (1, 2):
        slack = tm("yolo11n", 640, th) - fixed
        be["%s_%dthr" % (cond, th)] = (pv["bytes_mean"] * 8 / (slack / 1000) / 1e6) if slack > 0 else None
out["break_even_mbps"] = be
print("break-even Mbit/s:", {k: (round(v, 1) if v else None) for k, v in be.items()})

# relative drops
drops = {}
for m, v, s in (("yolo11n", "raw", 640), ("yolov8n", "raw", 640), ("yolo11s", "raw", 640), ("yolo11m", "raw", 640),
                ("yolo11m", "r640_q70", 640), ("yolo11m", "r640_q30", 640), ("yolo11m", "r480_q70", 480), ("yolo11s", "r640_q30", 640)):
    d, n = acc(m, "day_" + v, s), acc(m, "night_" + v, s)
    drops["%s %s" % (m, v)] = (round(d, 3), round(n, 3), round(100 * (n - d) / d, 1))
out["night_drop"] = drops
for k, v in drops.items(): print("%-22s day %.3f night %.3f change %+.1f%%" % (k, *v))
json.dump(out, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)

# Figure 1: accuracy vs transmitted size
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, cond, title in zip(axes, ("day", "night"), ("а) день", "б) ночь (имитация)")):
    for m, mk, col in (("yolo11s", "o", "#1f3864"), ("yolo11m", "s", "#b8562a")):
        for res, ls in ((640, "-"), (480, "--"), (320, ":")):
            qs = (10, 30, 50, 70, 90) if res == 640 else (30, 70)
            xs = [kb("%s_r%d_q%d" % (cond, res, q)) for q in qs]
            ys = [acc(m, "%s_r%d_q%d" % (cond, res, q), res) for q in qs]
            ax.plot(xs, ys, ls, marker=mk, ms=3.5, color=col, lw=1,
                    label="%s, %d px" % (m.replace("yolo", "YOLO"), res) if cond == "day" else None)
    ax.axhline(acc("yolo11n", cond + "_raw", 640), color="gray", lw=1, ls="-.")
    ax.text(4.3, acc("yolo11n", cond + "_raw", 640) + (0.01 if cond == "day" else -0.04), "YOLO11n на борту, 640 px", fontsize=8, color="gray")
    ax.set_xscale("log"); ax.set_xlim(4, 110)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("размер кадра, КБ (лог. шкала)"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.5)
axes[0].set_ylabel("mAP@0.5:0.95")
axes[0].legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_acc_vs_size.png"), dpi=300); plt.close(fig)

# Figure 2: end-to-end latency vs uplink
import numpy as np
bw = np.linspace(2, 100, 200)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for m, res, q, col, lab in (("yolo11s", 640, 70, "#1f3864", "узел: YOLO11s, 640 px, JPEG 70"),
                             ("yolo11s", 480, 70, "#4a78b8", "узел: YOLO11s, 480 px, JPEG 70"),
                             ("yolo11m", 480, 70, "#b8562a", "узел: YOLO11m, 480 px, JPEG 70")):
    ax.plot(bw, [e2e(m, "day", res, q, b) for b in bw], color=col, lw=1.3, label=lab)
for th, ls in ((1, ":"), (2, "--")):
    ax.axhline(tm("yolo11n", 640, th), color="gray", ls=ls, lw=1, label="на борту: YOLO11n, 640 px, %d поток%s" % (th, "" if th == 1 else "а"))
ax.set_ylim(0, 260); ax.set_xlabel("скорость канала машина-сервер, Мбит/с"); ax.set_ylabel("задержка на кадр, мс")
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_latency_vs_bw.png"), dpi=300); plt.close(fig)
print("figures ok")

# Figure 1: accuracy vs transmitted size
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, cond, title in zip(axes, ("day", "night"), ("(a) day", "(b) night (simulated)")):
    for m, mk, col in (("yolo11s", "o", "#1f3864"), ("yolo11m", "s", "#b8562a")):
        for res, ls in ((640, "-"), (480, "--"), (320, ":")):
            qs = (10, 30, 50, 70, 90) if res == 640 else (30, 70)
            xs = [kb("%s_r%d_q%d" % (cond, res, q)) for q in qs]
            ys = [acc(m, "%s_r%d_q%d" % (cond, res, q), res) for q in qs]
            ax.plot(xs, ys, ls, marker=mk, ms=3.5, color=col, lw=1,
                    label="%s, %d px" % (m.replace("yolo", "YOLO"), res) if cond == "day" else None)
    ax.axhline(acc("yolo11n", cond + "_raw", 640), color="gray", lw=1, ls="-.")
    ax.text(4.3, acc("yolo11n", cond + "_raw", 640) + (0.01 if cond == "day" else -0.04), "YOLO11n on board, 640 px", fontsize=8, color="gray")
    ax.set_xscale("log"); ax.set_xlim(4, 110)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("frame size, KB (log scale)"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.5)
axes[0].set_ylabel("mAP@0.5:0.95")
axes[0].legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_acc_vs_size_en.png"), dpi=300); plt.close(fig)

# Figure 2: end-to-end latency vs uplink
import numpy as np
bw = np.linspace(2, 100, 200)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for m, res, q, col, lab in (("yolo11s", 640, 70, "#1f3864", "roadside: YOLO11s, 640 px, JPEG 70"),
                             ("yolo11s", 480, 70, "#4a78b8", "roadside: YOLO11s, 480 px, JPEG 70"),
                             ("yolo11m", 480, 70, "#b8562a", "roadside: YOLO11m, 480 px, JPEG 70")):
    ax.plot(bw, [e2e(m, "day", res, q, b) for b in bw], color=col, lw=1.3, label=lab)
for th, ls in ((1, ":"), (2, "--")):
    ax.axhline(tm("yolo11n", 640, th), color="gray", ls=ls, lw=1, label="on board: YOLO11n, 640 px, %d thread%s" % (th, "" if th == 1 else "s"))
ax.set_ylim(0, 260); ax.set_xlabel("uplink bandwidth, Mbit/s"); ax.set_ylabel("latency per frame, ms")
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_latency_vs_bw_en.png"), dpi=300); plt.close(fig)
print("figures ok")
