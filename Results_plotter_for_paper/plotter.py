import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# === 1. Load & clean ===
df = pd.read_csv("data.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"Unnamed: 0": "workload"})

# Convert GPU‐memory columns to numeric (coerce errors → NaN) and to GB
methods = ["Actual GPU memory", "GPUMemNet", "FakeTensor", "Horus"]
for m in methods:
    df[m] = pd.to_numeric(df[m], errors="coerce") / 1000

# Fill down blank workloads, stringify batch sizes
df["workload"] = df["workload"].replace("", np.nan).ffill()
df["batch_size"] = df["batch_size"].astype(int).astype(str)

# Build a MultiIndex of (workload, batch_size)
tuples = list(zip(df["workload"], df["batch_size"]))
df = df.set_index(pd.MultiIndex.from_tuples(tuples, names=["Workload", "Batch Size"]))

# === 2. Reorder: workload groups, then sorted batch sizes ===
grouped = defaultdict(list)
for i, (wl, bs) in enumerate(df.index):
    grouped[wl].append((int(bs), i))

new_idx = []
for wl in grouped:
    for _, idx in sorted(grouped[wl], key=lambda x: x[0]):
        new_idx.append(idx)

df = df.iloc[new_idx]
group_labels = df.index.tolist()
x = np.arange(len(group_labels))

# === 3. Plot in GB ===
plt.figure(figsize=(15, 4))
ax = plt.gca()

bar_w = 0.2
colors = ["black", "dimgray", "darkgray", "lightgray"]
ymax = 45.0   # in GB

# draw each method
for i, m in enumerate(methods):
    vals = df[m].values
    pos = x + (i - 1.5) * bar_w
    ax.bar(pos, np.nan_to_num(vals), bar_w,
           color=colors[i], edgecolor="black", linewidth=0.8,
           label=m)
    for xi, v in zip(pos, vals):
        if np.isnan(v):
            ax.text(xi, 2.0, "X", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color="red")
        elif v > ymax:
            ax.text(xi, ymax - 20, f"{v:.1f}GB", ha="center",
                    va="bottom", rotation=90, fontsize=9,
                    fontweight="bold", color="black")

# --- Add vertical separators between workloads ---
prev_wl = group_labels[0][0]
boundaries = []
for idx, (wl, _) in enumerate(group_labels):
    if wl != prev_wl:
        boundaries.append(idx - 0.5)
        prev_wl = wl

for b in boundaries:
    ax.annotate(
        "",                        # no text
        xy=(b, 1),                 # at the top of the axes (fraction 1)
        xycoords=('data','axes fraction'),
        xytext=(b, -0.3),          # extend down into the margin (fraction -0.3)
        textcoords=('data','axes fraction'),
        arrowprops=dict(           # this draws just a line
            arrowstyle='-',
            linestyle='--',
            linewidth=1,
            color='black'
        ),
        clip_on=False
    )

# 40 GB reference line
ax.axhline(40, ls="--", color="black", lw=1.5)
ax.text(0, 41, "A100 40 GB Memory", ha="left", va="bottom",
        fontsize=10, color="black", fontweight="bold")

# bottom level: batch sizes tick labels
ax.set_xticks(x)
ax.set_xticklabels([bs for _, bs in group_labels],
                   rotation=0, fontsize=14)

# ▶️ place the 'Batch Size' title BELOW the workload labels:
ax.set_xlabel("Batch Size, Workload (from top to bottom)", fontsize=18, labelpad=90)   # <-- bumped labelpad

ax.set_ylabel("GPU Memory (GB)", fontsize=18)
ax.set_ylim(-1, ymax)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.legend(loc="upper center")

# top level: workload names on a twin axis
work_pos, work_labels = [], []
prev, start = None, 0
for i, (wl, _) in enumerate(group_labels + [(None,None)]):
    if wl != prev:
        if prev is not None:
            work_pos.append((start + i - 1) / 2)
            work_labels.append(prev)
        start = i
        prev = wl

ax_workload = ax.twiny()
ax_workload.set_frame_on(False)
ax_workload.xaxis.set_ticks_position("bottom")
ax_workload.xaxis.set_label_position("bottom")
ax_workload.spines["bottom"].set_position(("outward", 30))
ax_workload.set_xlim(ax.get_xlim())
ax_workload.set_xticks(work_pos)
ax_workload.set_xticklabels(work_labels,
                            fontsize=12, fontweight="bold",
                            rotation=50)
ax_workload.tick_params(length=0, pad=-5)

ax.tick_params(axis='y', labelsize=14)
plt.tight_layout(pad=1.0)
plt.savefig("test.pdf", pad_inches=0.2, bbox_inches="tight")