import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# === 1) Load & clean ===
df = pd.read_csv("data3.csv")

df.columns = [c.strip() for c in df.columns]
if "workload" not in df.columns:
    df = df.rename(columns={df.columns[0]: "workload"})

for col in ["Execution Time (h)", "GPU Memory (GB)"]:
    df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")

# ⚠️ Do NOT drop rows — we keep workloads even if all values are NaN
df["workload"] = df["workload"].astype(str)

x_labels = df["workload"].tolist()
x = np.arange(len(x_labels))

mem_vals = df["GPU Memory (GB)"].to_numpy(dtype=float)
time_vals = df["Execution Time (h)"].to_numpy(dtype=float)

# === 2) Plot ===
plt.figure(figsize=(9, 4.5))
ax_mem = plt.gca()          # GPU Memory → left axis
ax_time = ax_mem.twinx()    # Execution Time → right axis

# --- Colors ---
color_mem = "DarkRed"
color_time = "IndianRed"

# === GPU Memory (bars, left y-axis) ===
bar_w = 0.5
ymax_mem = max(40.0, np.nanmax(mem_vals[~np.isnan(mem_vals)]) if np.any(~np.isnan(mem_vals)) else 0)
ymax_mem = math.ceil((ymax_mem + 2) / 5.0) * 5.0 if ymax_mem > 0 else 10

bars = ax_mem.bar(
    x, np.nan_to_num(mem_vals, nan=0.0),
    width=bar_w, color=color_mem,
    edgecolor="black", linewidth=0.8,
    label="GPU Memory (GB)"
)

for xi, v in zip(x, mem_vals):
    if np.isnan(v):
        pass
        # Red X above baseline
        ax_mem.text(xi, 7.0, "X", ha="center", va="bottom",
                    fontsize=18, fontweight="bold", color="red")
    elif v > ymax_mem:
        ax_mem.text(xi, ymax_mem*0.9, f"{v:.1f} GB",
                    ha="center", va="top", rotation=90, fontsize=9,
                    fontweight="bold", color="black")

# 40 GB reference line
ax_mem.axhline(40, ls="--", color="black", lw=1.5)
ax_mem.text(0.02, 40 + ymax_mem*0.02, "A100 40 GB Memory",
            ha="left", va="bottom", fontsize=14, color="black", fontweight="bold")

# Format left axis
ax_mem.set_ylim(0, ymax_mem)
ax_mem.set_ylabel("GPU Memory (GB)", fontsize=18, color=color_mem)
ax_mem.tick_params(axis='y', labelcolor=color_mem, labelsize=18)

# === Execution Time (line, right y-axis) ===
ax_time.plot(
    x, time_vals,
    marker="o", linewidth=2, color=color_time,
    label="Execution Time (h)",
)

# Mark missing execution time with red OOM
if np.any(~np.isnan(time_vals)):
    ymax_time = np.nanmax(time_vals)
    ymax_time = (math.ceil((ymax_time + 0.2) / 0.5) * 0.5) if ymax_time < 10 else (math.ceil((ymax_time + 1) / 2) * 2)
else:
    ymax_time = 1.0
ax_time.set_ylim(0, max(1.0, ymax_time))

# (Fixed) apply OOM per missing time point
for xi, v in zip(x, time_vals):
    if np.isnan(v):
        ax_time.text(xi, 0.2, "OOM", ha="center", va="bottom",
                     fontsize=18, fontweight="bold", color="red")

# Format right axis
ax_time.set_ylabel("Execution Time (h)", fontsize=18, color=color_time)
ax_time.tick_params(axis='y', labelcolor=color_time, labelsize=18)

# === X axis ===
ax_mem.set_xticks(x)
ax_mem.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=18)

# === Legend (combine both) ===
handles = [bars, ax_time.lines[0]]
labels = ["GPU Memory (GB)", "Execution Time (h)"]
ax_mem.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.85), ncol=1, fontsize=12)

# === NEW: value annotations (added only) ===
# Memory values above bars
for xi, v in zip(x, mem_vals):
    if not np.isnan(v) and v >= 0:
        ax_mem.text(xi, v + max(0.01 * ymax_mem, 0.2), f"{v:.1f}",
                    ha="center", va="bottom", fontsize=18, color=color_mem, fontweight="bold")

# Time values near markers
for xi, v in zip(x, time_vals):
    if not np.isnan(v) and v >= 0:
        ax_time.text(xi, v + max(0.02 * ymax_time, 0.05), f"{v:.1f}",
                     ha="center", va="bottom", fontsize=18, color=color_time, fontweight="bold")

plt.tight_layout()
plt.savefig("poster-collocation-motiv.pdf", bbox_inches="tight", pad_inches=0.2)
plt.close()
