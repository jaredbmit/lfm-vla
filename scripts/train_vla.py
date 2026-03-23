"""VLA: VLM + learnable action query token + MLP action head, trained with BC on CALVIN."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from calvin_dataset import ACTION_TOKEN, CALVINDataset, make_calvin_collate_fn

CALVIN_BASE = "/home/jared/drl/calvin/dataset/calvin_debug_dataset"
MODEL_ID = "LiquidAI/LFM2-VL-3B"
SYSTEM_PROMPT = (
    "You are a robot manipulation agent. Given an image of the current scene "
    "and a language instruction, predict the next action to execute."
)

BATCH_SIZE = 4
NUM_STEPS = 1000
LOG_EVERY = 50
EVAL_EVERY = 200
LR = 1e-5
HIDDEN_DIM = 2048
ACTION_DIM = 7
CHUNK_SIZE = 10
MAX_LENGTH = 256


class VLA(nn.Module):
    def __init__(self, vlm, action_token_id: int, hidden_dim=HIDDEN_DIM,
                 action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.vlm = vlm
        self.action_token_id = action_token_id
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # VLM4VLA-style action head: hidden_dim -> 1024 -> 1024 -> chunk_size * action_dim
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, chunk_size * action_dim),
        )

    def forward(self, **vlm_inputs):
        outputs = self.vlm(**vlm_inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)

        # Find <action> token position in each sequence
        action_mask = vlm_inputs["input_ids"] == self.action_token_id  # (B, seq_len)
        action_idx = action_mask.long().argmax(dim=1)  # (B,) — first occurrence
        action_hidden = last_hidden[torch.arange(last_hidden.shape[0]), action_idx]  # (B, hidden_dim)

        raw = self.action_head(action_hidden.float())  # (B, chunk_size * action_dim)
        return raw.view(-1, self.chunk_size, self.action_dim)  # (B, chunk_size, 7)


def main():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_image_tokens=256)
    processor.tokenizer.padding_side = "right"

    # Add action query token to vocabulary
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})

    print("Loading VLM...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16,
    )
    vlm.resize_token_embeddings(len(processor.tokenizer))

    device = next(vlm.parameters()).device
    action_token_id = processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
    vla = VLA(vlm, action_token_id=action_token_id, chunk_size=CHUNK_SIZE).to(device)

    trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vla.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

    # Data
    train_ds = CALVINDataset(f"{CALVIN_BASE}/training", chunk_size=CHUNK_SIZE)
    val_ds = CALVINDataset(f"{CALVIN_BASE}/validation", chunk_size=CHUNK_SIZE)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    collate_fn = make_calvin_collate_fn(processor, SYSTEM_PROMPT, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(vla.parameters(), lr=LR)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    def loss_fn(pred, gt):
        # pred/gt: (B, chunk_size, 7)
        pose_loss = mse_fn(pred[:, :, :6], gt[:, :, :6])
        gripper_target = (gt[:, :, 6] + 1) / 2  # -1 -> 0, +1 -> 1
        gripper_loss = bce_fn(pred[:, :, 6], gripper_target)
        return pose_loss + gripper_loss

    print(f"\nTraining: {len(train_ds)} samples, {len(val_ds)} val, "
          f"{NUM_STEPS} steps, chunk_size={CHUNK_SIZE}\n")

    vla.train()
    train_iter = iter(train_loader)
    running_loss = 0.0

    for step in range(1, NUM_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        gt_actions = batch.pop("gt_actions").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        pred_actions = vla(**batch)
        loss = loss_fn(pred_actions, gt_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            print(f"  step {step:5d}  train_loss={avg_loss:.6f}")
            running_loss = 0.0

        if step % EVAL_EVERY == 0:
            vla.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    gt = val_batch.pop("gt_actions").to(device)
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    val_loss_sum += loss_fn(vla(**val_batch), gt).item()
                    val_steps += 1
            print(f"           val_loss={val_loss_sum / val_steps:.6f}")
            vla.train()

    # --- Final check ---
    vla.eval()
    batch = next(iter(val_loader))
    gt_actions = batch.pop("gt_actions").to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        pred_actions = vla(**batch)

    print(f"\nSample predictions (first action in chunk) vs ground truth:")
    for i in range(min(4, pred_actions.shape[0])):
        pred = pred_actions[i, 0].cpu().tolist()
        gt = gt_actions[i, 0].cpu().tolist()
        print(f"  [{i}] pred: [{', '.join(f'{x:+.3f}' for x in pred)}]")
        print(f"      gt:   [{', '.join(f'{x:+.3f}' for x in gt)}]")


if __name__ == "__main__":
    main()
