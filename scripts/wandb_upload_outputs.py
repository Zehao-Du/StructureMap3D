#!/usr/bin/env python3
"""
Upload everything under `outputs/` to Weights & Biases as versioned artifacts.

Usage examples:
  # Dry run - show what would be uploaded
  python scripts/wandb_upload_outputs.py --outputs outputs --project StructureMap3D-outputs --dry-run

  # Actual upload (assumes WANDB_API_KEY is configured in environment)
  python scripts/wandb_upload_outputs.py --outputs outputs --project StructureMap3D-outputs --entity your-wandb-entity

Notes:
- Each immediate subdirectory under `outputs/` will be uploaded as a separate artifact named `outputs/<subdir>`.
- You can set --max-size to skip folders larger than that size (in MB).
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import shutil
import re

try:
    import wandb
except Exception as e:
    print("ERROR: wandb is not installed. Install it with `pip install wandb` and configure WANDB_API_KEY.")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upload_outputs")


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Pi{suffix}"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def upload_folder(folder: Path, outputs_base: Path, project: str, entity: str | None, dry_run: bool, max_size_mb: int):
    """Upload a single run folder as an artifact. `outputs_base` is the base outputs path used to compute a safe artifact name."""
    # compute size and potentially skip large folders
    size_bytes = dir_size_bytes(folder)
    size_mb = size_bytes / (1024 * 1024)

    if max_size_mb and size_mb > max_size_mb:
        log.warning("Skipping %s (%.1f MB) because it exceeds --max-size=%d MB", folder, size_mb, max_size_mb)
        return

    log.info("Preparing: %s (size=%s, files=%d)", folder, sizeof_fmt(size_bytes), sum(1 for _ in folder.rglob('*') if _.is_file()))

    # construct a safe artifact name: outputs_<relpath> with non-allowed chars replaced by _
    try:
        rel = folder.relative_to(outputs_base)
    except Exception:
        rel = folder
    artifact_raw = f"outputs_{rel}"
    # replace path separators and any illegal characters
    artifact_name = str(artifact_raw).replace(os.sep, "_")
    artifact_name = re.sub(r"[^A-Za-z0-9_.-]", "_", artifact_name)

    if dry_run:
        log.info("Dry run: would upload artifact '%s' from folder '%s'", artifact_name, folder)
        return

    run_name = f"upload_{folder.name}"
    run = wandb.init(project=project, entity=entity, job_type="upload_outputs", name=run_name, reinit=True)
    art = wandb.Artifact(name=str(artifact_name), type="outputs")

    # Add directory contents to the artifact (store relative path under outputs)
    log.info("Adding directory to artifact: %s", folder)
    art.add_dir(str(folder), name=str(rel))

    log.info("Logging artifact '%s'...", artifact_name)
    run.log_artifact(art)
    run.finish()
    log.info("Uploaded artifact '%s' successfully.", artifact_name)


def main(argv):
    p = argparse.ArgumentParser(description="Upload 'outputs/' contents to Weights & Biases as artifacts.")
    p.add_argument("--outputs", default="outputs", help="Path to outputs directory (default: outputs)")
    p.add_argument("--project", required=True, help="wandb project name to upload into")
    p.add_argument("--entity", default=None, help="wandb entity (user or org). If not set, uses configured account.")
    p.add_argument("--dry-run", default=False, action="store_true", help="Do not upload, just show what would be done")
    p.add_argument("--max-size", default=0, type=int, help="Skip folders larger than this size in MB (0 = no limit)")
    p.add_argument("--pattern", default=None, help="Only upload folders whose names contain this pattern")

    args = p.parse_args(argv)

    outputs_path = Path(args.outputs).resolve()
    if not outputs_path.exists():
        log.error("Path does not exist: %s", outputs_path)
        sys.exit(2)

    # Identify run-level directories (prefer timestamped run folders like 2026-02-05_14-49-54)
    timestamp_re = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    run_roots = set()
    for sub in outputs_path.rglob('*'):
        if not sub.is_dir():
            continue
        if not any(p for p in sub.rglob('*') if p.is_file()):
            continue
        # climb ancestors to find an ancestor whose name looks like a timestamp run folder
        found = False
        for ancestor in [sub] + list(sub.parents):
            try:
                rel = ancestor.relative_to(outputs_path)
            except Exception:
                continue
            if timestamp_re.match(ancestor.name):
                run_roots.add(ancestor)
                found = True
                break
        if not found:
            # fallback: use first 3 path components under outputs if available
            rel = sub.relative_to(outputs_path)
            parts = rel.parts
            if len(parts) >= 3:
                cand = outputs_path.joinpath(*parts[:3])
            else:
                cand = outputs_path.joinpath(*parts)
            run_roots.add(cand)

    candidates = sorted(run_roots)

    # Filter by name pattern if provided
    if args.pattern:
        candidates = [c for c in candidates if args.pattern in c.name]

    if not candidates:
        log.warning("No candidate run folders found under %s", outputs_path)
        return

    log.info("Found %d candidate run folders to upload", len(candidates))

    for folder in candidates:
        upload_folder(folder, outputs_path, project=args.project, entity=args.entity, dry_run=args.dry_run, max_size_mb=args.max_size)


if __name__ == '__main__':
    main(sys.argv[1:])
