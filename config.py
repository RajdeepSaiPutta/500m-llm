# model config

class config:
    # model size
    vocab_size = 32_000
    n_layers = 24
    n_heads = 16
    d_model = 1280
    d_ff = 5120
    max_seq_len = 512
    dropout = 0.1

    # training
    batch_size = 8
    grad_accum = 32
    max_steps = 4_000_000
    lr = 3e-4
    warmup_steps = 5_000
    save_every = 50_000

    # data
    dataset = "huggingfacefw/fineweb-edu"
    data_sample = "sample-10bt"
    tokenizer_path = "./tokenizer"
    checkpoint_dir = "./checkpoints500m"
