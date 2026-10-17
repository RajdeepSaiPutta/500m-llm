# 500m llm

a 500 million parameter language model trained from scratch on fineweb edu. it has 24 layers, 16 attention heads, an embedding size of 1280 and a context of 512 tokens.

the model code is in model/gpt.py, the settings are in config.py, train.py trains it and rag_chat.py runs a small rag chat on top of it.
