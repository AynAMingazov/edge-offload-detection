# -*- coding: utf-8 -*-
"""Fig. 3: accuracy versus end-to-end latency of on-board and offloaded configurations (uplink 20 Mbit/s).
Also writes results/deploy.json with the numbers used in the paper."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f)))
A, T, Q, TQ, EX = L("results/acc.json"), L("results/time.json"), L("results/acc_q.json"), L("results/time_q.json"), L("results/acc_exdark.json")
PR = L("results/frames.json")["variants"]
PX = L("results/exdark_frames.json")["variants"]
BW = 20.0
tx = lambda b: b * 8 / (BW * 1e6) * 1000
t = lambda k: T[k]["total_ms_median"]
tq = lambda k: TQ[k]["total_ms_median"]
cod = lambda v: PR["day_" + v]["enc_ms"] + PR["day_" + v]["dec_ms"]

cfg = [  # label key, latency day, latency exdark, acc day, acc exdark, on board?
    ("n_fp32", t("yolo11n|640|1"), t("yolo11n|640|1"), A["yolo11n|day_raw|640"]["map"], EX["yolo11n|raw|640"]["map"], True),
    ("n_int8", tq("yolo11n_int8|640|1"), tq("yolo11n_int8|640|1"), Q["yolo11n_int8|day_raw|640"]["map"], Q["yolo11n_int8|exdark_raw|640"]["map"], True),
    ("s_int8", tq("yolo11s_int8|640|2"), tq("yolo11s_int8|640|2"), Q["yolo11s_int8|day_raw|640"]["map"], Q["yolo11s_int8|exdark_raw|640"]["map"], True),
    ("off_s", cod("r640_q70") + t("yolo11s|640|8") + tx(PR["day_r640_q70"]["bytes_mean"]),
     cod("r640_q70") + t("yolo11s|640|8") + tx(PX["r640_q70"]["bytes_mean"]),
     A["yolo11s|day_r640_q70|640"]["map"], EX["yolo11s|r640_q70|640"]["map"], False),
    ("off_s8", cod("r640_q70") + tq("yolo11s_int8|640|8") + tx(PR["day_r640_q70"]["bytes_mean"]),
     cod("r640_q70") + tq("yolo11s_int8|640|8") + tx(PX["r640_q70"]["bytes_mean"]),
     Q["yolo11s_int8|day_r640_q70|640"]["map"], Q["yolo11s_int8|exdark_r640_q70|640"]["map"], False),
    ("off_m", cod("r640_q90") + t("yolo11m|640|8") + tx(PR["day_r640_q90"]["bytes_mean"]),
     cod("r640_q90") + t("yolo11m|640|8") + tx(PX["r640_q90"]["bytes_mean"]),
     A["yolo11m|day_r640_q90|640"]["map"], EX["yolo11m|r640_q90|640"]["map"], False),
]
LAB = {
    "ru": {"n_fp32": "борт: YOLO11n FP32, 1 поток", "n_int8": "борт: YOLO11n INT8, 1 поток", "s_int8": "борт: YOLO11s INT8, 2 потока",
           "off_s": "узел: YOLO11s FP32, JPEG 70", "off_s8": "узел: YOLO11s INT8, JPEG 70", "off_m": "узел: YOLO11m FP32, JPEG 90",
           "pan": ("а) COCO, день", "б) ExDark, реальная низкая освещённость"), "x": "задержка на кадр, мс (канал 20 Мбит/с)", "y": "mAP@0.5:0.95"},
    "en": {"n_fp32": "on board: YOLO11n FP32, 1 thread", "n_int8": "on board: YOLO11n INT8, 1 thread", "s_int8": "on board: YOLO11s INT8, 2 threads",
           "off_s": "roadside: YOLO11s FP32, JPEG 70", "off_s8": "roadside: YOLO11s INT8, JPEG 70", "off_m": "roadside: YOLO11m FP32, JPEG 90",
           "pan": ("(a) COCO, daylight", "(b) ExDark, real low light"), "x": "latency per frame, ms (uplink 20 Mbit/s)", "y": "mAP@0.5:0.95"},
}
MK = {"n_fp32": "o", "n_int8": "s", "s_int8": "D", "off_s": "^", "off_s8": "v", "off_m": "P"}
for lang, suf in (("ru", ""), ("en", "_en")):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.0), dpi=200)
    for i, ax in enumerate(axs):
        for key, ld, le, ad, ae, ob in cfg:
            x, y = (ld, ad) if i == 0 else (le, ae)
            ax.scatter(x, y, marker=MK[key], s=55, color="black" if ob else "white", edgecolors="black", zorder=3,
                       label=LAB[lang][key])
        ax.set_title(LAB[lang]["pan"][i], fontsize=10); ax.set_xlabel(LAB[lang]["x"], fontsize=9)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=8); ax.set_xlim(0, 230)
        if i == 0:
            ax.set_ylabel(LAB[lang]["y"], fontsize=9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(HERE, "fig", "fig3_acc_vs_latency%s.png" % suf)); plt.close(fig)
json.dump({k: {"lat_day": ld, "lat_exdark": le, "acc_day": ad, "acc_exdark": ae} for k, ld, le, ad, ae, ob in cfg},
          open(os.path.join(HERE, "results", "deploy.json"), "w"), indent=1)
for k, ld, le, ad, ae, ob in cfg:
    print("%-7s %6.1f %6.1f  %.3f %.3f" % (k, ld, le, ad, ae))
