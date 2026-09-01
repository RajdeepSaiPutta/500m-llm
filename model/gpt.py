# this is a 500m parameter gpt model
# designed to be small and fast

import torch
import torch.nn as nn

# multi-head attention block
class multiheadattention(nn.module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        # qkv projection
        self.qkv = nn.linear(cfg.d_model, 3 * cfg.d_model, bias=false)
        self.out = nn.linear(cfg.d_model, cfg.d_model, bias=false)
        self.drop = nn.dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)

        def reshape(t):
            return t.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        # use built-in attention for speed
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=true,
            dropout_p=self.drop.p if self.training else 0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)

# feed forward block
class feedforward(nn.module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.sequential(
            nn.linear(cfg.d_model, cfg.d_ff, bias=false),
            nn.gelu(),
            nn.linear(cfg.d_ff, cfg.d_model, bias=false),
            nn.dropout(cfg.dropout)
        )

    def forward(self, x):
        return self.net(x)

# one transformer block
class transformerblock(nn.module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = multiheadattention(cfg)
        self.ff = feedforward(cfg)
        self.ln1 = nn.rmsnorm(cfg.d_model)
        self.ln2 = nn.rmsnorm(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# the main gpt model
class gpt(nn.module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.modulelist([transformerblock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.rmsnorm(cfg.d_model)
        self.lm_head = nn.linear(cfg.d_model, cfg.vocab_size, bias=false)
        # tie weights
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=none):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = none
        if targets is not none:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
