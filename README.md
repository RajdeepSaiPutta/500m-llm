# 500m llm

a small gpt model with 500 million parameters.

## model

- 24 layers
- 16 attention heads
- 1280 embedding dimension
- 512 max sequence length

## files

- `model/gpt.py` - the model code
- `config.py` - hyperparameters
- `train.py` - training script
- `rag_chat.py` - rag chatbot

## requirements

- pytorch
- transformers
- datasets
- tokenizers

## quick start

```bash
python rag_chat.py
```

## train

```bash
python train.py
```
