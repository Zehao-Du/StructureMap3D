#!/usr/bin/env python3
"""Simple utility to open an H5 file and recursively traverse its
contents, dumping groups and dataset values (or their shapes).

Usage::

    python read_h5.py /path/to/file.h5

The script builds a nested Python dictionary representing the file's
hierarchy.  Groups become ``dict``s and datasets are converted to
NumPy arrays (or, for large arrays, their shape and dtype are shown).
It also prints the structure with indentation so you can inspect keys
and contents at a glance.

This is useful when you want to explore the contents of a ManiSkill
trajectory file or any other HDF5 artifact.
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Any, Dict, Union

import h5py
import numpy as np


def recursively_read(h5obj: Union[h5py.File, h5py.Group, h5py.Dataset]) -> Any:
    """Convert an HDF5 object into a native Python structure.

    * Groups are turned into ``dict``s whose values are the result of a
      recursive call.  
    * Datasets are read into NumPy arrays with ``[:]``.  For very large
      datasets the caller may decide to inspect only ``.shape``/``.dtype``
      instead of materialising the entire array.
    """
    if isinstance(h5obj, h5py.Dataset):
        return h5obj[()]
    elif isinstance(h5obj, (h5py.File, h5py.Group)):
        out: Dict[str, Any] = {}
        for key, item in h5obj.items():
            out[key] = recursively_read(item)
        return out
    else:
        # unexpected type, just return None
        return None


def print_structure(
    obj: Union[Dict[str, Any], Any],
    indent: int = 0,
    max_array_elements: int = 10,
) -> None:
    """Pretty-print a nested dictionary produced by :func:`recursively_read`.

    * Groups (dicts) are shown with a trailing ``/`` and their children
      indented.
    * NumPy arrays are printed with their ``shape`` and ``dtype``.  If
      the total number of elements is small we also show the actual
      contents.
    * Other objects are printed using ``repr``.
    """
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{pad}{k}/")
                print_structure(v, indent + 1, max_array_elements)
            else:
                # dataset
                if isinstance(v, np.ndarray):
                    info = f"{v.shape}, {v.dtype}"
                    if v.size <= max_array_elements:
                        info += f", data={v.tolist()}"
                    print(f"{pad}{k}: {info}")
                else:
                    print(f"{pad}{k}: {repr(v)}")
    else:
        # root was not dict; just print it
        print(pad + repr(obj))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read H5 file and recursively dump keys/values."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the .h5 file to inspect",
    )
    parser.add_argument(
        "--print-dict",
        action="store_true",
        help="If set, print the Python dictionary representation after reading."
        .replace("\n", " "),
    )
    args = parser.parse_args()

    with h5py.File(args.file, "r") as hf:
        data = recursively_read(hf)

    print(f"Structure of {args.file}:")
    print_structure(data)

    if args.print_dict:
        import pprint

        pprint.pprint(data)


if __name__ == "__main__":
    main()
