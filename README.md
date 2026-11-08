# 500m llm

a 500 million parameter language model trained from scratch on fineweb edu. 24 layers, 16 attention heads, embedding size 1280, context 512 tokens.

## run it

### training

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. download fineweb-edu (or use your own dataset)
python data/download_fineweb.py

# 3. train the model
python train.py

# checkpoints saved to checkpoints/
```

### chat / rag

```bash
# run rag chat on top of trained model
python rag_chat.py
```

### model architecture

the model code is in `model/gpt.py`:
- 24 transformer layers
- 16 attention heads
- 1280 embedding dimension
- 512 context window
- SwiGLU activation, RoPE embeddings
- ~500M parameters

### config

settings in `config.py`:
```python
batch_size = 32
learning_rate = 3e-4
max_iters = 100000
eval_interval = 1000
```

---

*weights available at: https://huggingface.co/rajdeepp11/500mWithRAG-fromscratch*
