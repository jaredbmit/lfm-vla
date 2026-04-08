"""Inspect CALVIN action distributions and the VLM4VLA normalization transform.

Loads all rel_actions from a CALVIN split, visualizes raw distributions, and
shows the effect of clip-then-rescale normalization with different bounds.

Usage:
    uv run python experiments/inspect_action_normalization.py \
        [--dataset-dir /path/to/calvin/training]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def load_all_actions(dataset_dir: Path) -> np.ndarray:
    """Load rel_actions from every episode .npz file in the dataset."""
    npz_files = sorted(dataset_dir.rglob("episode_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No episode_*.npz files found under {dataset_dir}")
    actions = []
    for f in npz_files:
        ep = np.load(f)
        actions.append(ep["rel_actions"])
    return np.stack(actions)  # (N, 7)


def normalize_action(action: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    """VLM4VLA's normalize_action with maintain_last=True (gripper untouched)."""
    last_val = action[..., -1].copy()
    clipped = np.clip(action, a_min=norm_min, a_max=norm_max)
    normed = 2 * (clipped - norm_min) / (norm_max - norm_min) - 1
    normed[..., -1] = last_val
    return normed


def unnormalize_action(action: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    """VLM4VLA's unnoramalize_action with maintain_last=False (full reverse)."""
    last_val = action[..., -1].copy()
    res = 0.5 * (action + 1) * (norm_max - norm_min) + norm_min
    res[..., -1] = last_val
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",
                        default="/home/jared/drl/calvin/dataset/calvin_debug_dataset/training",
                        help="Path to a CALVIN training split with episode npz files")
    parser.add_argument("--out-dir", default="experiments/results",
                        help="Directory to save plots")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading actions from {dataset_dir} ...")
    raw = load_all_actions(dataset_dir)
    N = raw.shape[0]
    print(f"Loaded {N:,} actions, shape {raw.shape}")

    # ── Summary statistics ───────────────────────────────────────────────────
    pose = raw[:, :6]
    gripper = raw[:, 6]

    print("\n--- Raw action statistics (dims 0-5: pose, dim 6: gripper) ---")
    print(f"{'dim':<8} {'mean':>10} {'std':>10} {'min':>10} {'p1':>10} {'p5':>10} {'p95':>10} {'p99':>10} {'max':>10}")
    for d in range(7):
        col = raw[:, d]
        print(f"{DIM_NAMES[d]:<8} {col.mean():10.5f} {col.std():10.5f} "
              f"{col.min():10.5f} {np.percentile(col, 1):10.5f} {np.percentile(col, 5):10.5f} "
              f"{np.percentile(col, 95):10.5f} {np.percentile(col, 99):10.5f} {col.max():10.5f}")

    # ── Fraction clipped at various bounds ───────────────────────────────────
    print("\n--- Fraction of pose values clipped at different bounds ---")
    bounds = [0.3, 0.5, 0.65, 0.8, 1.0, 1.5, 2.0]
    for b in bounds:
        frac = np.mean(np.abs(pose) > b)
        print(f"  |action| > {b:.2f}:  {frac * 100:.3f}%  ({int(frac * pose.size):,} / {pose.size:,} values)")

    # ── Apply VLM4VLA normalization ──────────────────────────────────────────
    NORM_MIN, NORM_MAX = -0.65, 0.65
    normed = normalize_action(raw, NORM_MIN, NORM_MAX)
    roundtrip = unnormalize_action(normed, NORM_MIN, NORM_MAX)

    # Round-trip error (only for values within clip bounds)
    within_bounds = np.all((raw[:, :6] >= NORM_MIN) & (raw[:, :6] <= NORM_MAX), axis=1)
    if within_bounds.sum() > 0:
        rt_err = np.abs(raw[within_bounds, :6] - roundtrip[within_bounds, :6])
        print(f"\nRound-trip error (within-bounds samples): max={rt_err.max():.2e}, mean={rt_err.mean():.2e}")

    clipped_mask = ~within_bounds
    print(f"Samples with at least one clipped pose dim: {clipped_mask.sum():,} / {N:,} ({clipped_mask.mean()*100:.2f}%)")

    # ── Clipping error analysis ──────────────────────────────────────────────
    clip_error = raw[:, :6] - roundtrip[:, :6]
    print(f"\n--- Clipping error per dim (raw - roundtrip, pose only) ---")
    print(f"{'dim':<8} {'mean_abs':>10} {'max_abs':>10} {'frac>0':>10}")
    for d in range(6):
        err = clip_error[:, d]
        abs_err = np.abs(err)
        print(f"{DIM_NAMES[d]:<8} {abs_err.mean():10.6f} {abs_err.max():10.5f} {(abs_err > 0).mean()*100:9.3f}%")

    # ── PLOTS ────────────────────────────────────────────────────────────────

    # 1. Raw histograms per dimension
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flat
    for d in range(7):
        ax = axes[d]
        col = raw[:, d]
        ax.hist(col, bins=200, color="steelblue", alpha=0.8, edgecolor="none")
        ax.axvline(NORM_MIN, color="red", ls="--", lw=1, label=f"clip={NORM_MIN}")
        ax.axvline(NORM_MAX, color="red", ls="--", lw=1, label=f"clip={NORM_MAX}")
        ax.set_title(f"Raw {DIM_NAMES[d]}")
        ax.set_ylabel("count")
        if d == 0:
            ax.legend(fontsize=8)
    axes[7].axis("off")
    fig.suptitle("CALVIN rel_actions — Raw Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "action_raw_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'action_raw_histograms.png'}")

    # 2. Normalized histograms per dimension
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flat
    for d in range(7):
        ax = axes[d]
        col = normed[:, d]
        ax.hist(col, bins=200, color="darkorange", alpha=0.8, edgecolor="none")
        ax.axvline(-1, color="gray", ls=":", lw=1)
        ax.axvline(1, color="gray", ls=":", lw=1)
        ax.set_title(f"Normalized {DIM_NAMES[d]}")
        ax.set_ylabel("count")
        ax.set_xlim(-1.15, 1.15)
    axes[7].axis("off")
    fig.suptitle(f"CALVIN rel_actions — After VLM4VLA Normalization (clip [{NORM_MIN}, {NORM_MAX}] → [-1, 1])",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "action_normalized_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'action_normalized_histograms.png'}")

    # 3. Overlay: raw vs normalized (pose dims only, shared axes per dim)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flat
    for d in range(6):
        ax = axes[d]
        ax.hist(raw[:, d], bins=200, color="steelblue", alpha=0.6, label="raw", density=True)
        ax.hist(normed[:, d], bins=200, color="darkorange", alpha=0.6, label="normalized", density=True)
        ax.axvline(NORM_MIN, color="red", ls="--", lw=1)
        ax.axvline(NORM_MAX, color="red", ls="--", lw=1)
        ax.set_title(DIM_NAMES[d])
        ax.set_ylabel("density")
        if d == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Raw vs Normalized Pose Actions (density)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "action_raw_vs_normalized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'action_raw_vs_normalized.png'}")

    # 4. Log-scale tails: CDF of |action| to see tail behavior
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0, 2.0, 500)
    for d in range(6):
        fracs = [np.mean(np.abs(raw[:, d]) > t) for t in thresholds]
        ax.plot(thresholds, fracs, label=DIM_NAMES[d], lw=1.5)
    ax.axvline(0.65, color="red", ls="--", lw=1, label="clip bound (0.65)")
    ax.set_xlabel("|action value|")
    ax.set_ylabel("P(|action| > threshold)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_title("Tail Distribution of Raw Pose Actions", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "action_tail_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'action_tail_cdf.png'}")

    # 5. Scatter: Tanh output range vs raw action range
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flat
    for d in range(6):
        ax = axes[d]
        sorted_raw = np.sort(raw[:, d])
        sorted_normed = np.sort(normed[:, d])
        # Subsample for scatter performance
        step = max(1, len(sorted_raw) // 5000)
        ax.scatter(sorted_raw[::step], sorted_normed[::step], s=1, alpha=0.3, color="teal")
        ax.axhline(-1, color="gray", ls=":", lw=0.5)
        ax.axhline(1, color="gray", ls=":", lw=0.5)
        ax.axvline(NORM_MIN, color="red", ls="--", lw=0.8)
        ax.axvline(NORM_MAX, color="red", ls="--", lw=0.8)
        ax.set_xlabel("raw action")
        ax.set_ylabel("normalized action")
        ax.set_title(DIM_NAMES[d])
    fig.suptitle("Mapping: Raw → Normalized (per-dim sorted scatter)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "action_mapping_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'action_mapping_scatter.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
