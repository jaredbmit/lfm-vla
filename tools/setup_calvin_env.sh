#!/bin/bash
# Sets up the calvin_venv conda environment for CALVIN simulation + eval client.
# Run this once from the repo root:
#   bash scripts/setup_calvin_env.sh ~/drl/calvin
#
# Requires: conda, git

set -euo pipefail

CALVIN_DIR="${1:-$HOME/drl/calvin}"

if [ ! -d "$CALVIN_DIR" ]; then
    echo "Error: CALVIN directory not found at $CALVIN_DIR"
    echo "Usage: bash scripts/setup_calvin_env.sh <path/to/calvin>"
    exit 1
fi

echo "==> Creating conda environment calvin_venv (Python 3.8)..."
conda create -n calvin_venv python=3.8 -y

echo "==> Installing CALVIN packages..."
conda run -n calvin_venv bash -c "
    cd '$CALVIN_DIR'
    pip install wheel cmake==3.18.4
    cd calvin_env/tacto && pip install -e . && cd ..
    pip install -e .
    cd ../calvin_models && pip install -e .
"

echo "==> Applying NumPy 1.24 compatibility patches..."
PYTHON_SITE=$(conda run -n calvin_venv python -c "import site; print(site.getsitepackages()[0])")

# networkx: np.int removed in NumPy 1.24
sed -i 's/(np\.int, "int")/(np.int_, "int")/g' \
    "$PYTHON_SITE/networkx/readwrite/graphml.py"

# urdfpy: np.float removed in NumPy 1.24
sed -i 's/\.astype(np\.float)/.astype(np.float64)/g' \
    "$PYTHON_SITE/urdfpy/urdf.py"

# tacto (vendored in calvin_env): np.float removed in NumPy 1.24
sed -i 's/color\.astype(np\.float)/color.astype(np.float64)/g' \
    "$CALVIN_DIR/calvin_env/tacto/tacto/renderer.py"

echo "==> Done. Activate with: conda activate calvin_venv"
