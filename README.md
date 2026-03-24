# lfm-vla

Fine-tuning LFM2-VL into a VLA policy, evaluated on the [CALVIN](https://github.com/mees/calvin) benchmark.

The project is split into two isolated environments to avoid dependency conflicts between the legacy CALVIN simulator (Python 3.8, PyTorch 1.13) and the modern training stack (Python 3.13, PyTorch 2.x).

```
Terminal 1 (calvin_venv)          Terminal 2 (.venv)
┌─────────────────────────┐       ┌──────────────────────────┐
│  eval_client.py         │       │  eval_server.py          │
│  - CALVIN pybullet sim  │ JSON  │  - loads VLA checkpoint  │
│  - steps env            │ <---> │  - runs model forward    │
│  - records videos       │ TCP   │  - returns action chunks │
└─────────────────────────┘       └──────────────────────────┘
```

---

## Installation

### 1. Training environment (eval server + training)

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <this repo>
cd lfm-vla
uv sync
```

### 2. CALVIN simulation environment (eval client)

Requires conda. Clone CALVIN alongside this repo first:

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git ~/drl/calvin
```

Then run the setup script (creates `calvin_venv` and applies NumPy 1.24 compatibility patches):

```bash
bash scripts/setup_calvin_env.sh ~/drl/calvin
```

Download a dataset (debug dataset is sufficient for smoke-testing):

```bash
cd ~/drl/calvin
bash dataset/download_data.sh
```

---

## Running an evaluation

Start both processes in separate terminals. The client will wait up to 5 minutes for the server to become ready, so order doesn't strictly matter.

**Terminal 1 — inference server** (runs in `.venv`):

```bash
uv run python scripts/eval_server.py \
    --checkpoint runs/<timestamp>/checkpoints/best \
    --port 5555
```

**Terminal 2 — CALVIN eval client** (runs in `calvin_venv`):

```bash
conda activate calvin_venv
python scripts/eval_client.py \
    --dataset_path ~/drl/calvin/dataset/task_D_D \
    --num_sequences 1000 \
    --num_videos 10 \
    --eval_log_dir runs/eval_001
```

Results are printed as success rates across 1–5 chained tasks. Videos (MP4) are saved to `runs/eval_001/videos/` for the first `--num_videos` sequences, with the current instruction burned into each frame.

### Eval client options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_path` | required | Path to CALVIN dataset split |
| `--num_sequences` | 1000 | Number of evaluation sequences |
| `--eval_log_dir` | None | Directory for results + videos |
| `--num_videos` | 0 | Save MP4s for the first N sequences |
| `--video_dir` | `eval_log_dir/videos` | Override video output directory |
| `--host` | localhost | Inference server host |
| `--port` | 5555 | Inference server port |
| `--debug` | False | Verbose per-step output |

---

## Training

```bash
uv run python scripts/train.py
```

Checkpoints are saved to `runs/<timestamp>/checkpoints/`.

---

## Protocol

`eval_client.py` and `eval_server.py` communicate over a persistent TCP connection using newline-delimited JSON.

**Request** (client → server):
```json
{"image": "<base64-encoded PNG>", "instruction": "open the drawer"}
```

**Response** (server → client):
```json
{"actions": [[0.01, -0.02, 0.0, 0.0, 0.01, 0.0, 1.0], ...]}
```

Each response contains `CHUNK_SIZE` actions (7-dim: 6 pose deltas + gripper ∈ {-1, 1}). The client executes all actions before requesting the next chunk.

**Shutdown** (client → server, sent after all sequences complete):
```json
{"shutdown": true}
```
