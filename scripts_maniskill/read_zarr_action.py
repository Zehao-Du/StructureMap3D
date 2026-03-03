#!/usr/bin/env python3
"""Simple utility to open a Zarr store and report information about an
"action" dataset or group.

Usage:
    python read_zarr_action.py /path/to/store.zarr

The script will try to locate an entry named ``action`` within the store.
If it is an array the shape and dtype are printed.  If it is a group the
contents of the group are iterated and each array's shape/dtype displayed.
"""

import argparse
import sys

import zarr


def describe_array(name, arr):
    try:
        shape = arr.shape
        dtype = arr.dtype
    except Exception:
        shape = None
        dtype = None
    print(f"  {name}: shape={shape}, dtype={dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Read a Zarr store and print information about its 'action' entry."
    )
    parser.add_argument(
        "store",
        help="Path to the Zarr store (directory or file) to open",
    )
    args = parser.parse_args()

    try:
        root = zarr.open(args.store, mode="r")
    except Exception as e:
        print(f"failed to open store '{args.store}': {e}", file=sys.stderr)
        sys.exit(1)

    # the dataset we care about lives under data/actions
    try:
        arr = root["data"]["actions"]
    except Exception:
        print("could not locate data/actions array in the store", file=sys.stderr)
        sys.exit(1)

    if not isinstance(arr, zarr.core.Array):
        print("data/actions is not an array", file=sys.stderr)
        sys.exit(1)

    # print the shape and dtype of the actions array
    print(f"actions: shape={arr.shape}, dtype={arr.dtype}")

    # compute per-channel statistics assuming second dimension is channels
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        data = arr[:]  # load into memory
        # expect a 2‑D array where second dimension is channel count
        if data.ndim == 2 and data.shape[1] == 7:
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            mins = np.min(data, axis=0)
            maxs = np.max(data, axis=0)
            print("per-channel statistics:")
            for i in range(data.shape[1]):
                print(
                    f"  channel {i}: mean={means[i]}, std={stds[i]}, "
                    f"min={mins[i]}, max={maxs[i]}"
                )

            # if we have at least 7 channels, sample unique values from the 7th
            if data.shape[1] >= 7:
                ch7 = data[:, 6]
                unique_vals = np.unique(ch7)
                if unique_vals.size > 0:
                    # choose up to 5 non-repeating samples
                    samples = unique_vals[:5] if unique_vals.size >= 5 else unique_vals
                    print("five non-repeating values from channel 7:")
                    for v in samples:
                        print(f"  {v}")
                else:
                    print("channel 7 contains no values")
        else:
            print("unexpected actions shape for channel stats, skipping")
    else:
        print("numpy not available, cannot compute statistics")


if __name__ == "__main__":
    main()
