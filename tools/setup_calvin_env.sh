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

echo "==> Patching calvin_models requirements (drop unbuildable pyhash)..."
# pyhash 0.9.3 can no longer build on current PyPI: its setup.py uses
# setuptools' removed `use_2to3` and its setup_requires pins cmake==3.18.4,
# which was yanked. We stub it below instead.
sed -i '/^pyhash$/d' "$CALVIN_DIR/calvin_models/requirements.txt"

echo "==> Installing CALVIN packages..."
conda run -n calvin_venv bash -c "
    cd '$CALVIN_DIR'
    pip install wheel cmake
    cd calvin_env/tacto && pip install -e . && cd ..
    pip install -e .
    cd ../calvin_models && pip install -e .
"

echo "==> Installing pyhash stub (fnv1_32 only)..."
PYTHON_SITE=$(conda run -n calvin_venv python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$PYTHON_SITE/pyhash"
cat > "$PYTHON_SITE/pyhash/__init__.py" <<'PYEOF'
"""Minimal pyhash stub: only fnv1_32 is used by calvin_agent."""

class fnv1_32:
    def __call__(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        h = 0x811c9dc5
        for b in data:
            h = ((h * 0x01000193) ^ b) & 0xffffffff
        return h
PYEOF

echo "==> Applying NumPy 1.24 compatibility patches..."

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
