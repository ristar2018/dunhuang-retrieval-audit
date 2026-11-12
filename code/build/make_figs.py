# -*- coding: utf-8 -*-
"""Build manuscript figures 1-5 at journal column widths.

Sources (all under the stage-0 review directory):
  e1_2_dose.csv                  dose-response, two arms
  e1_2b_error_sensitivity.csv    error-rate scan
  e1_eval.csv                    five-fold protocol (450 runs x best/final)
  e1_1_main_tests.csv            exploratory Holm family (10 contrasts)
  e0_2_eval.csv                  margin/scale factorial on S0 (variants B and F)
  e1_3_perclass.csv              per-class AP by tier

Run:  python make_figs.py     ->  figs/fig{1..5}.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

MM = 1 / 25.4
W2 = 174 * MM          # double-column width
W1 = 84 * MM           # single-column width

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.dpi": 300, "figure.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
RED, PURPLE, SKY = "#D55E00", "#CC79A7", "#56B4E9"
GREY, DARK = "#666666", "#222222"

D = r"<PROJECT_ROOT>\08_阶段0复核"
OUT = os.path.join(D, "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("->", name)


# ---------------------------------------------------------------- figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 64 * MM))
    ax.set_xlim(-3.5, 103.5); ax.set_ylim(-1, 113); ax.axis("off")

    def box(x0, y0, x1, y1, text, fc="#FFFFFF", fs=7):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=0.6, edgecolor=DARK, facecolor=fc))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=fs, color=DARK, linespacing=1.35)

    def arrow(p, q, ls="-", color=DARK, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7,
                                     linewidth=0.6, color=color, linestyle=ls,
                                     shrinkA=0.5, shrinkB=0.5))

    yt, yb = 80, 34                       # top and bottom row centres
    h = 12                                # half box height
    box(0, yt - h, 20, yt + h, "Image corpus\n32,415 images", fc="#F4F4F4")
    box(26, yt - h, 49, yt + h, "k-NN label\npropagation ($k$ = 7)")
    box(55, yt - h, 77, yt + h, "Training labels\n$L_0$–$L_3$", fc="#FDF0DC")
    box(83, yt - h, 100, yt + h, "Projection head\n(frozen backbone)")
    box(55, yb - h, 77, yb + h, "Gallery answer key\n339 human / 21,703", fc="#FDF0DC")
    box(83, yb - h, 100, yb + h, "MAP")

    arrow((20, yt), (26, yt))
    arrow((49, yt), (55, yt))
    arrow((77, yt), (83, yt))
    arrow((44.5, yt - h), (44.5, yb))
    arrow((44.5, yb), (55, yb))
    arrow((77, yb), (83, yb))
    arrow((91.5, yt - h), (91.5, yb + h))

    ax.annotate("", xy=(66, yt - h), xytext=(66, yb + h),
                arrowprops=dict(arrowstyle="<|-|>", linewidth=0.6, color=RED,
                                linestyle=(0, (2, 1.5)), mutation_scale=7))
    ax.text(67.5, (yt + yb) / 2, "one pass,\none provenance", color=RED, fontsize=7,
            ha="left", va="center", linespacing=1.35)

    ax.text(0, 2, "18,000 gallery candidates are flagged as distractors yet retain a category label;\n"
                  "under the original answer key those labels define correct retrievals.",
            fontsize=7, color=DARK, ha="left", va="bottom", linespacing=1.5)
    save(fig, "fig1_provenance_loop")


# ---------------------------------------------------------------- figure 2
def fig2():
    dose = pd.read_csv(os.path.join(D, "e1_2_dose.csv"), encoding="utf-8-sig")
    sens = pd.read_csv(os.path.join(D, "e1_2b_error_sensitivity.csv"), encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 68 * MM),
                             gridspec_kw=dict(wspace=0.28, width_ratios=[1.08, 1]))

    # --- (a) dose-response: paired delta at each pool size
    ax = axes[0]
    piv = dose.pivot_table(index=["method", "seed", "arm", "n_p"],
                           columns="dataset", values="MAP")
    delta = (piv["S2"] - piv["S0"]).groupby(level=["method", "arm", "n_p"]).mean()
    nps = [500, 1000, 2000, 4000, 8000, 17000]
    styles = {"ArcFace": (BLUE, "o"), "GCE": (ORANGE, "s")}
    for method, (col, mk) in styles.items():
        base = delta.loc[(method, "G_clean", 0)]
        for arm, ls, lab in (("G_knn", "-", "labels admitted"),
                             ("G_null", "--", "labels withheld")):
            ys = [base] + [delta.loc[(method, arm, n)] for n in nps]
            ax.plot([0] + nps, ys, ls=ls, color=col, marker=mk, ms=3, lw=1.0,
                    mfc=col, mec=col, label="%s, %s" % (method, lab))
    ax.axhline(0, color=GREY, lw=0.5, ls=":")
    ax.set_xscale("symlog", linthresh=500)
    ax.set_xticks([0] + nps)
    ax.set_xticklabels(["0", "0.5", "1", "2", "4", "8", "17"])
    ax.set_xlabel("Propagated images admitted ($\\times 10^{3}$)")
    ax.set_ylabel("$\\Delta$MAP  (S2 $-$ S0, paired)")
    ax.set_title("(a) Same images, two answer keys", loc="left")
    ax.set_ylim(-0.20, 0.06)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.1,
              labelspacing=0.25)

    # --- (b) error-rate scan
    ax = axes[1]
    cols = {"ArcFace": BLUE, "GCE": ORANGE, "SCE": GREEN, "SubCenterArcFace_K3": PURPLE}
    names = {"ArcFace": "ArcFace", "GCE": "GCE", "SCE": "SCE", "SubCenterArcFace_K3": "SC-K3"}
    piv2 = sens.pivot_table(index=["method", "seed", "rho"], columns="dataset", values="MAP")
    for method, col in cols.items():
        d = (piv2.loc[method, "S2"] - piv2.loc[method, "S0"]).groupby("rho").mean()
        ax.plot(d.index, d.values, color=col, lw=1.0, label=names[method])
        cross = float(np.interp(0.0, d.values[::-1], d.index.values[::-1]))
        ax.plot([cross], [0], marker="|", color=col, ms=6, mew=1.0)
    ax.axhline(0, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("Share of propagated labels replaced ($\\rho$)")
    ax.set_ylabel("$\\Delta$MAP  (S2 $-$ S0)")
    ax.set_title("(b) Where the sign would change", loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.028, 0.088)
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, borderpad=0.1,
              labelspacing=0.25, ncol=2, columnspacing=0.9,
              bbox_to_anchor=(-0.02, -0.02))
    ax.text(0.97, 0.95, "$\\rho^{*}$: 0.55–0.67", fontsize=7, color=DARK,
            ha="right", va="top", transform=ax.transAxes)

    save(fig, "fig2_dose_response")


# ---------------------------------------------------------------- figure 3
def fig3():
    ev = pd.read_csv(os.path.join(D, "e1_eval.csv"), encoding="utf-8-sig")
    ev = ev[ev.ckpt == "best"]
    tests = pd.read_csv(os.path.join(D, "e1_1_main_tests.csv"), encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 68 * MM),
                             gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.05]))

    # --- (a) clean-protocol curves
    ax = axes[0]
    series = [("ArcFace", "ArcFace", BLUE, "o"), ("GCE", "GCE", ORANGE, "s"),
              ("SCE", "SCE", GREEN, "^"), ("SubCenterArcFace_K3", "SC-K3", PURPLE, "D")]
    tiers = ["S0", "S1", "S2", "S3"]
    for key, lab, col, mk in series:
        m = ev[(ev.method == key) & (ev.tier.isin(tiers))].groupby("tier")["MAP"].mean()
        ax.plot(range(4), [m[t] for t in tiers], color=col, marker=mk, ms=3, lw=1.0,
                label=lab, mfc=col, mec=col)
    for key, col, mk in [("ArcFace", BLUE, "o"), ("GCE", ORANGE, "s")]:
        prim = ["S0_prime", "S1_prime", "S2_prime", "S3_prime"]
        m = ev[(ev.method == key) & (ev.tier.isin(prim))].groupby("tier")["MAP"].mean()
        ax.plot(range(4), [m[t] for t in prim], color=col, marker=mk, ms=3, lw=0.9,
                ls="--", mfc="white", mec=col)
    ax.set_xticks(range(4))
    ax.set_xticklabels(tiers)
    ax.set_xlabel("Confidence tier (cumulative training set)")
    ax.set_ylabel("micro-MAP under a human-labelled key")
    ax.set_title("(a) Five-fold cross-fitted protocol", loc="left")
    ax.set_ylim(0.518, 0.668)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderpad=0.1,
              labelspacing=0.25, ncol=2, columnspacing=0.9)
    ax.text(0.02, 0.02, "dashed: sample size held at |S0|", fontsize=7, color=GREY,
            ha="left", va="bottom", transform=ax.transAxes)

    # --- (b) forest plot of the pre-declared family
    ax = axes[1]
    order = tests.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(order))
    for i, r in order.iterrows():
        sig = bool(r["sig_holm"])
        col = GREEN if sig and r["mean_delta"] > 0 else (RED if sig else GREY)
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=col, lw=1.0, solid_capstyle="butt")
        ax.plot([r["mean_delta"]], [i], marker="o", ms=3.2, color=col, mfc=col, mec=col)
    ax.axvline(0, color=GREY, lw=0.5, ls=":")
    short = {"SubCenterArcFace_K3": "SC-K3", "ArcFace": "ArcFace",
             "GCE": "GCE", "SCE": "SCE"}
    lab = []
    for c in order["contrast"]:
        head, tail = c.split(": ")
        contrast = tail.split("\u0394")[-1]
        contrast = "".join(ch for ch in contrast
                           if ch not in "\u7b49\u6837\u672c\u91cf ()").strip()
        lab.append("%s %s" % (short[head], contrast))
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    split_at = [i for i, l in enumerate(lab) if "\u2032" in l]
    if split_at:
        ax.axhline(max(split_at) + 0.5, color=GREY, lw=0.5, ls=(0, (1, 1.5)))
    ax.set_xlabel("$\\Delta$ micro-MAP  (95% CI, $n$ = 5 folds)")
    ax.set_title("(b) Ten exploratory contrasts", loc="left")
    ax.set_xlim(-0.10, 0.10)
    ax.set_ylim(-0.7, len(order) + 0.2)
    for i, r in order.iterrows():
        ax.text(0.098, i, "%.3f" % r["p_holm"], fontsize=7, ha="right", va="center",
                color=DARK)
    ax.text(0.098, len(order) - 0.1, "$p_{\\mathrm{Holm}}$", fontsize=7, ha="right",
            va="center", color=DARK)

    save(fig, "fig3_clean_protocol")


# ---------------------------------------------------------------- figure 4
def fig4():
    ev = pd.read_csv(os.path.join(D, "e0_2_eval.csv"), encoding="utf-8-sig")
    ev = ev[(ev.ckpt == "best") & (ev.dataset == "S0")]

    rows = [("orig", "ArcFace", "ArcFace s30 0.5$^\\circ$"),
            ("e02a", "ArcFace", "ArcFace s30 28.6$^\\circ$"),
            ("e02b", "ArcFace", "ArcFace s64 28.6$^\\circ$"),
            ("orig", "SubCenterArcFace_K1", "SC K=1 s64 28.6$^\\circ$"),
            ("orig", "SubCenterArcFace_K3", "SC K=3 s64 28.6$^\\circ$"),
            ("orig", "GCE", "GCE s30")]
    labels, vF, vB = [], [], []
    for tag, method, lab in rows:
        sub = ev[(ev.tag == tag) & (ev.method == method)]
        labels.append(lab)
        vF.append(sub[sub.variant == "F_seed_all"]["MAP"].mean())
        vB.append(sub[sub.variant == "B_seed_only"]["MAP"].mean())

    fig, ax = plt.subplots(figsize=(W1 - 15 * MM, 66 * MM))
    y = np.arange(len(labels))[::-1]
    h = 0.34
    ax.barh(y + h / 2 + 0.02, vF, height=h, color=SKY, edgecolor=DARK, lw=0.4,
            label="1,941 human labels (F)")
    ax.barh(y - h / 2 - 0.02, vB, height=h, color=BLUE, edgecolor=DARK, lw=0.4,
            label="339 human labels (B)")
    for yy, v in zip(y + h / 2 + 0.02, vF):
        ax.text(v + 0.008, yy, "%.3f" % v, va="center", fontsize=7)
    for yy, v in zip(y - h / 2 - 0.02, vB):
        ax.text(v + 0.008, yy, "%.3f" % v, va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("MAP at S0 (human labels only), mean of 3 seeds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.15), ncol=2,
              frameon=False, handlelength=1.4, columnspacing=1.2, borderpad=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig4_margin_units")


# ---------------------------------------------------------------- figure 5
def fig5():
    pc = pd.read_csv(os.path.join(D, "e1_3_perclass.csv"), encoding="utf-8-sig")
    d = pc[pc.method == "ArcFace"].sort_values("AP_S0").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(W1 - 12 * MM, 72 * MM))
    y = np.arange(len(d))
    for i, r in d.iterrows():
        col = GREEN if r["delta_S2_S0"] > 0 else RED
        ax.plot([r["AP_S0"], r["AP_S2"]], [i, i], color=col, lw=0.8, alpha=0.55)
        ax.plot([r["AP_S0"]], [i], marker="o", ms=2.8, mfc="white", mec=DARK, mew=0.7)
        ax.plot([r["AP_S2"]], [i], marker="o", ms=3.0, color=col, mec=col)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s (%d)" % (r["label"], round(r["size"])) for _, r in d.iterrows()])
    ax.set_xlabel("Average precision\n(five-fold human-labelled key)")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.plot([], [], marker="o", ls="", mfc="white", mec=DARK, mew=0.7, ms=3, label="S0")
    ax.plot([], [], marker="o", ls="", color=GREEN, ms=3, label="S2, gain")
    ax.plot([], [], marker="o", ls="", color=RED, ms=3, label="S2, loss")
    ax.legend(loc="lower right", frameon=False, handlelength=1.0, borderpad=0.1,
              labelspacing=0.25)
    ax.text(0.02, 0.97, "Spearman $\\rho$ = 0.071, $p$ = 0.795\n(class size vs $\\Delta$AP)",
            transform=ax.transAxes, fontsize=7, color=GREY, va="top", linespacing=1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig5_per_class")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()