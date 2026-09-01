# train a 500m gpt model

import torch
import os
from torch.optim import AdamW
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from model.gpt import GPT
from config import Config
import math

cfg = Config()

# load tokenizer
tokenizer = ByteLevelBPETokenizer(
    "./tokenizer/vocab.json",
    "./tokenizer/merges.txt"
)

# load dataset
dataset = load_dataset(
    cfg.dataset, name=cfg.data_sample,
    split="train", streaming=True
)

# get batch from streaming data
def get_batch(data_iter, batch_size, seq_len):
    buffer = []
    for ex in data_iter:
        ids = tokenizer.encode(ex["text"]).ids
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len + 1:]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y
            if len(buffer) < seq_len + 1:
                break

# make model
model = GPT(cfg).cuda().to(torch.bfloat16)
print(f"model params: {model.count_params() / 1e6:.1f}m")

# optimizer
optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95))

# learning rate schedule
def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))

os.makedirs(cfg.checkpoint_dir, exist_ok=true)

# training loop
data_iter = iter(dataset)
batch_gen = get_batch(data_iter, cfg.batch_size, cfg.max_seq_len)

step = 0
accum_loss = 0

optimizer.zero_grad()

for x, y in batch_gen:
    if step >= cfg.max_steps:
        break

    x, y = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, loss = model(x, y)
        loss = loss / cfg.grad_accum

    loss.backward()
    accum_loss += loss.item()

    if (step + 1) % cfg.grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()

        print(f"step {step+1} | loss: {accum_loss:.4f} | lr: {lr:.2e}")
        accum_loss = 0

    if (step + 1) % cfg.save_every == 0:
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f"{cfg.checkpoint_dir}/ckpt_{step+1}.pt")
        print(f"saved at step {step+1}")

    step += 1
