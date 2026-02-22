"""Merge datasets produced by ``gen_data_maniskill_old.py --num-procs``.

Each worker creates a subdirectory ``proc_0``, ``proc_1`` etc containing one
or more ``*.zarr`` files.  This script concatenates the contents of those
files along the leading dimension and writes a new Zarr for each task.

Usage example::

    python stack_zarr.py /data2/lirui/StructureMap3D/data_new/maniskill \
        --output /data2/lirui/StructureMap3D/data_new/maniskill/merged

If ``--output`` is provided it is treated as a directory; otherwise the
merged files are written into ``<base_dir>/merged/<name>.zarr``.
"""

import argparse
import sys
import pathlib
import textwrap
from collections import defaultdict

import zarr
import numpy as np


def _concat_along_first_axis(datasets, out_ds, offsets=None):
    """Concatenate list of datasets into *out_ds*.

    Both *datasets* and *out_ds* are instances of :class:`zarr.core.Array`.
    We assume every member has the same ``shape[1:]`` and compatible dtype.
    If *offsets* is provided, it must be the same length as *datasets*.
    """
    total = 0
    for ds in datasets:
        if ds.ndim == 0:
            continue
        total += ds.shape[0]
    # resize output to hold everything
    if out_ds.ndim > 0:
        newshape = list(out_ds.shape)
        newshape[0] = total
        out_ds.resize(tuple(newshape))

    idx = 0
    for i, ds in enumerate(datasets):
        if ds.ndim == 0:
            # scalars; just copy last value
            out_ds[...] = ds[...]
            continue
        n = ds.shape[0]
        if n == 0:
            continue
        
        data = ds[:]
        if offsets is not None:
            data = data + offsets[i]
        out_ds[idx : idx + n] = data
        idx += n


def merge_group(paths, out_path):
    """Merge a list of ``*.zarr`` files (all with same basename) into one.

    ``paths`` should be sorted so that the output is deterministic.
    """
    print(f"Merging {len(paths)} files into {out_path}")

    # log shapes from each input file for verification
    for p in paths:
        g = zarr.open(str(p), mode="r")
        print(f"  source {p.name}:")
        for grp_name in g:
            grp = g[grp_name]
            for ds_name, ds in grp.items():
                print(f"    {grp_name}/{ds_name}: {ds.shape}")
    first = zarr.open(str(paths[0]), mode="r")

    # prepare output root
    out_root = zarr.open(str(out_path), mode="w")
    for grp_name in ("data", "meta"):
        if grp_name not in first:
            continue
        out_grp = out_root.create_group(grp_name)
        for ds_name, ds in first[grp_name].items():
            shape = list(ds.shape)
            if len(shape) >= 1:
                shape[0] = 0
            compressor = ds.compressor
            kwargs = {}
            # preserve object_codec if present; if dtype is object but codec
            # is missing, supply MsgPack manually so that zarr doesn't complain.
            if hasattr(ds, "object_codec") and ds.object_codec is not None:
                kwargs["object_codec"] = ds.object_codec
            elif ds.dtype == object:
                from numcodecs import MsgPack

                kwargs["object_codec"] = MsgPack()
            out_grp.create_dataset(
                ds_name,
                shape=tuple(shape),
                dtype=ds.dtype,
                chunks=ds.chunks,
                compressor=compressor,
                **kwargs,
            )

    # collect all source arrays for each dataset so we can concatenate
    sources = {"data": defaultdict(list), "meta": defaultdict(list)}
    frame_offsets = []
    current_offset = 0
    for p in paths:
        g = zarr.open(str(p), mode="r")
        
        # Determine number of frames in this file to compute offsets for episode_ends
        n_frames = 0
        if "data" in g:
            for ds in g["data"].values():
                if ds.ndim > 0:
                    n_frames = ds.shape[0]
                    break
        frame_offsets.append(current_offset)
        current_offset += n_frames
        
        for grp_name in ("data", "meta"):
            if grp_name not in g:
                continue
            for ds_name, ds in g[grp_name].items():
                sources[grp_name][ds_name].append(ds)

    # perform concatenation once per dataset
    for grp_name, ds_dict in sources.items():
        out_grp = out_root[grp_name]
        for ds_name, dss in ds_dict.items():
            # episode_ends contains indices into the flattened data array,
            # so we must add the preceding frame count to each one.
            if ds_name == "episode_ends":
                _concat_along_first_axis(dss, out_grp[ds_name], offsets=frame_offsets)
            else:
                _concat_along_first_axis(dss, out_grp[ds_name])

    # log resulting shapes
    print(f"  merged output {out_path.name} shapes:")
    for grp_name in out_root:
        grp = out_root[grp_name]
        for ds_name, ds in grp.items():
            print(f"    {grp_name}/{ds_name}: {ds.shape}")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge proc_*/<name>.zarr files produced by gen_data_maniskill_old.py"
    )
    parser.add_argument("base_dir", type=pathlib.Path, help="directory containing proc_* subdirectories")
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=None,
        help="directory where merged zarrs will be written (created if missing)",
    )
    args = parser.parse_args()

    base = args.base_dir
    if not base.is_dir():
        print(f"error: {base} is not a directory", file=sys.stderr)
        sys.exit(1)

    proc_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("proc_")])
    if not proc_dirs:
        print("no proc_* subdirectories found", file=sys.stderr)
        sys.exit(1)

    # collect files by basename
    groups = defaultdict(list)
    for d in proc_dirs:
        for z in d.glob("*.zarr"):
            groups[z.name].append(z)

    out_base = args.output or (base / "merged")
    out_base.mkdir(parents=True, exist_ok=True)

    for name, paths in groups.items():
        paths = sorted(paths)
        dest = out_base / name
        if dest.exists():
            print(f"warning: {dest} already exists, skipping")
            continue
        merge_group(paths, dest)

    print("All done")
