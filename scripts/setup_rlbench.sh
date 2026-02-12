#!/bin/bash
# Install RLBench for LIFT3D/MapPolicy when git submodule fails (e.g. SSH permission denied).
# Run from repo root: bash scripts/setup_rlbench.sh
#
# Prerequisite: CoppeliaSim and COPPELIASIM_ROOT (required by PyRep/RLBench).
#   1) Download: https://downloads.coppeliarobotics.com/ (e.g. CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)
#   2) Extract: mkdir -p $HOME/Programs/CoppeliaSim && tar -xf CoppeliaSim_Edu_*.tar.xz -C $HOME/Programs/CoppeliaSim --strip-components 1
#   3) Export:   export COPPELIASIM_ROOT=$HOME/Programs/CoppeliaSim
#   4) (optional) export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
#
# This script clones RLBench via HTTPS and runs pip install -e .

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIFT3D_ROOT="$REPO_ROOT/MapPolicy/models/LIFT3D"
THIRD_PARTY="$LIFT3D_ROOT/third_party"
RLBENCH_DIR="$THIRD_PARTY/RLBench"

echo "Repo root: $REPO_ROOT"
echo "Target:    $RLBENCH_DIR"

if [ -z "${COPPELIASIM_ROOT}" ]; then
    echo "Error: COPPELIASIM_ROOT is not set. PyRep (RLBench dependency) requires it to build."
    echo ""
    echo "Do this first:"
    echo "  1. Download CoppeliaSim Edu from https://downloads.coppeliarobotics.com/"
    echo "     (e.g. CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz for Ubuntu 20.04)"
    echo "  2. Extract and set the variable:"
    echo "     mkdir -p \$HOME/Programs/CoppeliaSim"
    echo "     tar -xf CoppeliaSim_Edu_*.tar.xz -C \$HOME/Programs/CoppeliaSim --strip-components 1"
    echo "     export COPPELIASIM_ROOT=\$HOME/Programs/CoppeliaSim"
    echo "  3. Re-run this script: bash scripts/setup_rlbench.sh"
    exit 1
fi
if [ ! -d "${COPPELIASIM_ROOT}" ]; then
    echo "Error: COPPELIASIM_ROOT is set to '$COPPELIASIM_ROOT' but that directory does not exist."
    exit 1
fi
echo "COPPELIASIM_ROOT=$COPPELIASIM_ROOT"

if [ ! -d "$LIFT3D_ROOT" ]; then
    echo "Error: LIFT3D not found at $LIFT3D_ROOT"
    exit 1
fi

mkdir -p "$THIRD_PARTY"
if [ -d "$RLBENCH_DIR" ]; then
    if [ ! -f "$RLBENCH_DIR/setup.py" ] && [ ! -f "$RLBENCH_DIR/pyproject.toml" ]; then
        echo "Removing incomplete/failed clone $RLBENCH_DIR"
        rm -rf "$RLBENCH_DIR"
    else
        echo "RLBench already present at $RLBENCH_DIR (has setup.py or pyproject.toml). Skipping clone."
        cd "$RLBENCH_DIR"
        export COPPELIASIM_ROOT  # ensure PyRep build sees it
        pip install -e .
        echo "Done."
        exit 0
    fi
fi

# Try LIFT3D fork via HTTPS first (branch lift3d)
echo "Cloning RLBench (CSCSX fork, HTTPS)..."
if git clone --depth 1 --branch lift3d https://github.com/CSCSX/RLBench.git "$RLBENCH_DIR" 2>/dev/null; then
    echo "Cloned CSCSX/RLBench branch lift3d."
else
    echo "CSCSX/RLBench clone failed (maybe private or no lift3d branch). Trying official stepjam/RLBench..."
    rm -rf "$RLBENCH_DIR"
    git clone --depth 1 https://github.com/stepjam/RLBench.git "$RLBENCH_DIR"
fi

cd "$RLBENCH_DIR"
if [ ! -f setup.py ] && [ ! -f pyproject.toml ]; then
    echo "Error: Clone completed but no setup.py or pyproject.toml in $RLBENCH_DIR"
    exit 1
fi
export COPPELIASIM_ROOT  # required for PyRep build
pip install -e .
echo "RLBench installed successfully. You can run: python scripts/gen_data_rlbench.py --task-name close_box --save-dir data_new/rlbench"
