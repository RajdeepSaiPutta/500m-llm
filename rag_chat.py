# rag chatbot for 500m model
# uses retrieval to help the small model answer questions

import os
import sys
import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from tokenizers import ByteLevelBPETokenizer

sys.path.append('/home/rajdeep/llm125')
from model.gpt import GPT
from config import Config

# try to load optional deps
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    use_semantic = True
except ImportError:
    use_semantic = False

# knowledge base stores facts and lets you search them
class knowledgebase:
    def __init__(self, storage_path="/home/rajdeep/llm125/documents"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=true)
        self.topics = {}
        self.documents = []
        self.embeddings = none
        self.faiss_index = none
        self.embedder = none
        self._load_default_topics()
        self._load_uploaded_docs()
        if use_semantic:
            self._build_index()

    # default topics to know about
    def _load_default_topics(self):
        defaults = {
            "machine learning": "ml lets computers learn from data. types: supervised (labeled data), unsupervised (no labels), reinforcement (rewards based).",
            "supervised learning": "uses labeled data to learn. examples: classification, regression. algorithms: linear regression, decision trees, neural nets.",
            "unsupervised learning": "finds patterns in unlabeled data. examples: clustering (k-means), dim reduction (pca), anomaly detection.",
            "reinforcement learning": "trains agents with rewards. used in robotics, games (alphago), self-driving.",
            "deep learning": "uses neural nets with many layers. archs: cnns (images), rnns (sequences), transformers (language).",
            "neural network": "layers of neurons that process info. connections have weights updated via backprop and gradient descent.",
            "backpropagation": "computes loss gradients w.r.t. weights. uses chain rule to propagate errors backward.",
            "transformer": "neural net using self-attention. key parts: multi-head attention, positional encoding, ff layers. powers gpt, bert.",
            "self-attention": "each token attends to all other tokens. computes q, k, v vectors. attention = softmax(qk_t/sqrt(d_k))v.",
            "multi-head attention": "runs many attention heads in parallel. each learns different relations. outputs concat and project.",
            "positional encoding": "adds position info to embeddings. uses sin/cos functions. lets model know token order.",
            "feed-forward network": "in transformers: two linear layers with gelu. expands to d_ff then projects back.",
            "bert": "encoder-only transformer. pretrained with masked lm and next sentence prediction. understands context both ways.",
            "gpt": "decoder-only transformer. pretrained with autoregressive lm. generates text by predicting next token.",
            "http": "transfers web data. methods: get (retrieve), post (create), put (update), delete (remove). https adds tls encryption.",
            "dns": "translates domains to ips. hier: root, tld, auth servers. uses caching for speed.",
            "cpu": "executes instructions. parts: control unit, alu, registers, cache (l1/l2/l3). modern cpus have multiple cores.",
            "gpu": "has thousands of cores for parallel work. key for deep learning, graphics. cuda enables general-purpose gpu coding.",
            "python": "high-level interpreted language. used in web (django, fastapi), data (pandas, numpy), ai (pytorch, tf), automation.",
            "docker": "containers package apps with deps. images are layered, share host kernel. same env across dev/staging/prod.",
            "kubernetes": "orchestrates containers. handles scheduling, scaling, load balancing, self-healing. uses pods, services, deployments.",
        }
        self.topics.update(defaults)
        for title, content in defaults.items():
            self.documents.append({"title": title, "content": content})

    # load user-uploaded docs
    def _load_uploaded_docs(self):
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key not in ['title', 'content'] and isinstance(value, str):
                                self.topics[key.lower()] = value
                                self.documents.append({"title": key, "content": value})
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'title' in item and 'content' in item:
                                self.topics[item['title'].lower()] = item['content']
                                self.documents.append(item)
            except exception:
                pass

    # build semantic search index
    def _build_index(self):
        try:
            self.embedder = SentenceTransformer("all-minilm-l6-v2")
            texts = [f"{d['title']}: {d['content']}" for d in self.documents]
            vectors = self.embedder.encode(texts, convert_to_numpy=true, normalize_embeddings=true)
            self.embeddings = vectors.astype(np.float32)
            self.faiss_index = faiss.indexflatip(self.embeddings.shape[1])
            self.faiss_index.add(self.embeddings)
            print(f"[ok] built index with {len(self.documents)} docs")
        except exception as e:
            print(f"[warn] semantic search unavailable: {e}")
            self.embedder = none

    # add a new doc
    def add_doc(self, title, content):
        self.topics[title.lower()] = content
        doc = {"title": title, "content": content}
        self.documents.append(doc)
        file_path = self.storage_path / f"{title.replace(' ', '_')}.json"
        with open(file_path, 'w') as f:
            json.dump(doc, f, indent=2)
        if use_semantic and self.embedder:
            vec = self.embedder.encode([f"{title}: {content}"], convert_to_numpy=true, normalize_embeddings=true)
            self.faiss_index.add(vec.astype(np.float32))
        print(f"[ok] added: {title}")

    # search docs
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if use_semantic and self.embedder and self.faiss_index is not none:
            return self._semantic_search(query, top_k)
        return self._keyword_search(query, top_k)

    # semantic search using embeddings
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        q_vec = self.embedder.encode([query], convert_to_numpy=true, normalize_embeddings=true).astype(np.float32)
        scores, indices = self.faiss_index.search(q_vec, min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < 0.2:
                continue
            results.append({
                'title': self.documents[idx]['title'],
                'content': self.documents[idx]['content'],
                'score': float(score)
            })
        return results

    # simple keyword search
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        query_lower = query.lower()
        scored = []
        for doc in self.documents:
            score = 0
            title_lower = doc['title'].lower()
            content_lower = doc['content'].lower()
            for word in query_lower.split():
                if len(word) > 2:
                    if word in title_lower:
                        score += 3
                    if word in content_lower:
                        score += 1
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=true)
        return [{'title': d['title'], 'content': d['content'], 'score': s} for s, d in scored[:top_k]]

    # list all known topics
    def list_topics(self):
        return sorted(self.topics.keys())

# manage conversation history
class conversationmanager:
    def __init__(self, max_history: int = 8, max_context_tokens: int = 1024):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        self.system_prompt = (
            "you are an ai/ml expert. terms like 'transformer', 'attention', 'neural network', "
            "'deep learning', 'machine learning' are ai concepts. "
            "use retrieved context as primary source. be accurate and concise."
        )

    def add_user(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def _trim(self):
        while len(self.history) > self.max_history * 2:
            self.history.pop(0)

    # build prompt with rag context
    def build_prompt(self, tokenizer, current_query: str, rag_context: Optional[str] = none) -> str:
        parts = [f"system: {self.system_prompt}"]

        if rag_context:
            parts.append(f"=== retrieved context (use this to answer) ===\n{rag_context}\n=== end context ===")
            parts.append("instruction: answer the user's question using only the context above. if the context doesn't have the answer, say 'i don't know'.")

        for msg in self.history:
            role = "user" if msg["role"] == "user" else "assistant"
            parts.append(f"{role}: {msg['content']}")

        parts.append(f"user: {current_query}")
        parts.append("assistant:")
        return "\n".join(parts)

    def clear(self):
        self.history = []

# load model and tokenizer
def load_model_and_tokenizer():
    cfg = Config()

    tokenizer = ByteLevelBPETokenizer(
        "/home/rajdeep/llm125/tokenizer/vocab.json",
        "/home/rajdeep/llm125/tokenizer/merges.txt"
    )
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")

    cfg.vocab_size = tokenizer.get_vocab_size()
    print(f"vocab size: {cfg.vocab_size}")

    checkpoint_path = '/home/rajdeep/llm125/checkpoints500m/ckpt_4000000.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = GPT(cfg)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=false)

    return model, tokenizer

# apply lora checkpoint
def apply_lora(model, lora_path: str):
    if not os.path.exists(lora_path):
        print(f"[warn] lora not found: {lora_path}")
        return model
    checkpoint = torch.load(lora_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=false)
    print(f"[ok] loaded lora from {lora_path}")
    return model

# clean up model output
def clean_response(text: str) -> str:
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'\?{3,}', '??', text)
    text = re.sub(r'!{3,}', '!!', text)
    text = re.sub(r'\s\w+-\s*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)

# generate text
@torch.no_grad()
def generate(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stop_sequences: Optional[List[str]] = none
) -> str:
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids]).to(device)

    # truncate if too long
    max_context = 512 - max_new_tokens - 10
    if input_ids.shape[1] > max_context:
        input_ids = input_ids[:, -max_context:]

    generated = input_ids.clone()

    if stop_sequences is none:
        stop_sequences = ["user:", "assistant:", "system:", "\n\nuser:", "\n\nassistant:"]

    seen_ngrams = set()

    for _ in range(max_new_tokens):
        if generated.shape[1] >= 512:
            break

        logits, _ = model(generated)
        logits = logits[:, -1, :] / temperature

        # repetition penalty
        if repetition_penalty != 1.0:
            vocab_size = logits.size(-1)
            for token_id in set(generated[0].tolist()):
                if token_id < vocab_size:
                    logits[0, token_id] /= repetition_penalty

        # top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=true)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            indices_to_remove = cum_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0
            logits[indices_to_remove.scatter(1, sorted_indices, indices_to_remove)] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        # stop on pad/eos/unk
        if token_id in [0, 2, 3]:
            break

        generated = torch.cat([generated, next_token], dim=1)

        # n-gram repetition check
        if generated.shape[1] >= 4:
            last_4 = tuple(generated[0, -4:].tolist())
            if last_4 in seen_ngrams:
                break
            seen_ngrams.add(last_4)

        # check stop sequences
        current_text = tokenizer.decode(generated[0, input_ids.shape[1]:].tolist())
        for phrase in stop_sequences:
            if phrase in current_text and len(current_text) > 20:
                return clean_response(current_text.split(phrase)[0])

    response = tokenizer.decode(generated[0, input_ids.shape[1]:].tolist())
    return clean_response(response)

# main chat function
def main():
    print("="*60)
    print("500m gpt rag chatbot")
    print("="*60)

    model, tokenizer = load_model_and_tokenizer()
    kb = knowledgebase()
    conv = conversationmanager()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # auto-load lora and enable rag
    model = apply_lora(model, './lora_manual_final.pt')
    use_rag = true

    print(f"device: {device}")
    print(f"docs: {len(kb.documents)}")
    print(f"semantic search: {'enabled' if use_semantic else 'disabled (keyword only)'}")
    print("="*60)
    print("\ncommands:")
    print("  list - show all docs")
    print("  rag on/off - toggle rag")
    print("  add <title> - add a doc")
    print("  clear - clear history")
    print("  temp <x> - set temp (0.1-2.0)")
    print("  lora <path> - load lora")
    print("  quit - exit")
    print("\n" + "="*60 + "\n")

    temperature = 0.7
    top_p = 0.9

    while true:
        try:
            user_input = input("> ").strip()
        except (eoferror, keyboardinterrupt):
            print("\nbye!")
            break

        if user_input.lower() == 'quit':
            print("bye!")
            break

        if user_input.lower() == 'list':
            topics = kb.list_topics()
            print(f"\n[docs] {len(topics)} docs:")
            for t in topics:
                print(f"  - {t}")
            print()
            continue

        if user_input.lower() == 'rag on':
            use_rag = true
            print("[ok] rag enabled\n")
            continue

        if user_input.lower() == 'rag off':
            use_rag = false
            print("[ok] rag disabled\n")
            continue

        if user_input.lower().startswith('add '):
            title = user_input[4:].strip()
            print(f"enter content for '{title}' (type 'end' on new line):")
            lines = []
            while true:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            kb.add_doc(title, '\n'.join(lines))
            print()
            continue

        if user_input.lower() == 'clear':
            conv.clear()
            print("[ok] history cleared\n")
            continue

        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input[5:])
                temperature = max(0.1, min(2.0, temperature))
                print(f"[ok] temp set to {temperature}\n")
            except:
                print("[err] bad temp\n")
            continue

        if user_input.lower().startswith('lora '):
            lora_path = user_input[5:].strip()
            model = apply_lora(model, lora_path)
            print()
            continue

        if not user_input:
            continue

        # rag retrieval
        rag_context = none
        docs = []
        if use_rag:
            docs = kb.search(user_input, top_k=3)
            if docs:
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    context_parts.append(f"[{i}] {doc['title']}: {doc['content']}")
                rag_context = "\n".join(context_parts)
                print(f"\n[rag] found {len(docs)} doc(s)")

        # direct rag answer for factual queries
        use_direct_rag = false
        direct_answer = none
        if docs and docs[0]['score'] > 0.4:
            factual_keywords = ['what is', 'what are', 'how does', 'how do', 'explain', 'define', 'describe']
            query_lower = user_input.lower()
            if any(kw in query_lower for kw in factual_keywords):
                query_words = set(query_lower.split())
                doc_words = set(docs[0]['content'].lower().split())
                overlap = len(query_words & doc_words) / max(len(query_words), 1)
                if overlap > 0.1 or docs[0]['score'] > 0.5:
                    use_direct_rag = true
                    direct_answer = docs[0]['content']
                    print(f"[ok] using direct rag answer (score: {docs[0]['score']:.2f})")

        if use_direct_rag and direct_answer:
            response = direct_answer
        else:
            # lower temp for model fallback
            fallback_temp = min(temperature, 0.5)
            fallback_top_p = min(top_p, 0.8)
            prompt = conv.build_prompt(tokenizer, user_input, rag_context)
            print("\n[thinking]...\n")
            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=200,
                temperature=fallback_temp,
                top_p=fallback_top_p,
                top_k=40,
                repetition_penalty=1.15
            )

        conv.add_user(user_input)
        if response:
            conv.add_assistant(response)
            print(f"{response}\n")
        else:
            print("[no response]\n")

if __name__ == "__main__":
    main()
