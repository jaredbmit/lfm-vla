"""Measure CALVIN episode segment lengths and quantify how many samples are
lost vs masked when using action chunking near segment boundaries.

Usage:
    python experiments/inspect_episode_lengths.py /path/to/task_ABC_D_annotated/training
    python experiments/inspect_episode_lengths.py /path/to/task_ABC_D_annotated/validation
"""

import argparse
from pathlib import Path

import numpy as np

CHUNK_SIZE = 10  # fwd_pred_next_n in VLM4VLA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", help="Path to training/ or validation/ split")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    chunk_size = args.chunk_size

    ann = np.load(
        split_dir / "lang_annotations" / "auto_lang_ann.npy",
        allow_pickle=True,
    ).item()

    indx = ann["info"]["indx"]  # list of (start_frame, end_frame)
    texts = ann["language"]["ann"]

    print(f"Split: {split_dir}")
    print(f"Annotated segments: {len(indx)}")
    print(f"Chunk size: {chunk_size}")
    print()

    lengths = []
    samples_ours = 0       # frames we keep (full chunks only)
    samples_vlm4vla = 0    # frames VLM4VLA keeps (partial chunks masked)
    partial_chunks = 0     # samples that would have incomplete chunks
    too_short = 0          # segments shorter than chunk_size

    for start, end in indx:
        seg_len = end - start + 1  # number of frames in segment
        lengths.append(seg_len)

        # Our approach: skip frames that can't form a full chunk
        # range(start, end + 1 - (chunk_size - 1)) => end - start + 2 - chunk_size frames
        n_full = max(0, seg_len - chunk_size + 1)
        samples_ours += n_full

        # VLM4VLA approach: include all frames, mask partial chunks
        # range(start_idx, end_idx + 1 - window_size) with window_size=1 => all frames
        n_all = seg_len
        samples_vlm4vla += n_all

        # Frames near the end that have partial chunks
        n_partial = seg_len - n_full
        partial_chunks += n_partial

        if seg_len < chunk_size:
            too_short += 1

    lengths = np.array(lengths)

    print("=== Segment Length Statistics ===")
    print(f"  Count:  {len(lengths)}")
    print(f"  Mean:   {lengths.mean():.1f}")
    print(f"  Std:    {lengths.std():.1f}")
    print(f"  Min:    {lengths.min()}")
    print(f"  P5:     {np.percentile(lengths, 5):.0f}")
    print(f"  P25:    {np.percentile(lengths, 25):.0f}")
    print(f"  Median: {np.median(lengths):.0f}")
    print(f"  P75:    {np.percentile(lengths, 75):.0f}")
    print(f"  P95:    {np.percentile(lengths, 95):.0f}")
    print(f"  Max:    {lengths.max()}")
    print()

    print(f"=== Impact on Training Samples (chunk_size={chunk_size}) ===")
    print(f"  Segments shorter than chunk_size: {too_short} / {len(lengths)} "
          f"({100*too_short/len(lengths):.1f}%)")
    print(f"  Samples (ours, full chunks only):     {samples_ours:,}")
    print(f"  Samples (VLM4VLA, partial + masked):  {samples_vlm4vla:,}")
    print(f"  Difference:                           {samples_vlm4vla - samples_ours:,} "
          f"({100*(samples_vlm4vla - samples_ours)/samples_vlm4vla:.1f}% of VLM4VLA total)")
    print(f"  Partial-chunk samples:                {partial_chunks:,} "
          f"({100*partial_chunks/samples_vlm4vla:.1f}% of VLM4VLA total)")
    print()

    # Histogram of segment lengths
    bins = [0, 10, 20, 30, 40, 50, 64, 80, 100, 150, 200, 500, 1000]
    print("=== Segment Length Distribution ===")
    print(f"  {'Bin':>12s}  {'Count':>6s}  {'%':>6s}  {'Lost frames':>12s}")
    for i in range(len(bins) - 1):
        mask = (lengths >= bins[i]) & (lengths < bins[i+1])
        count = mask.sum()
        if count == 0:
            continue
        # For each segment in this bin, lost frames = min(chunk_size-1, seg_len)
        lost = sum(min(chunk_size - 1, l) for l in lengths[mask])
        print(f"  [{bins[i]:>4d}, {bins[i+1]:>4d})  {count:>6d}  {100*count/len(lengths):>5.1f}%  {lost:>12,}")
    # overflow bin
    mask = lengths >= bins[-1]
    if mask.sum() > 0:
        lost = sum(min(chunk_size - 1, l) for l in lengths[mask])
        print(f"  [{bins[-1]:>4d},   +∞)  {mask.sum():>6d}  {100*mask.sum()/len(lengths):>5.1f}%  {lost:>12,}")


if __name__ == "__main__":
    main()
