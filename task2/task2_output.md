# 🗺️ Task 2: abctokz Architecture Map

This explains how the `abctokz` codebase is split up, why it's split that way, and where the code has some bugs. 

---

## 🛠️ Jobs: Who Does What?

### Job 1 — Training
**File:** `src/abctokz/trainers/bpe_trainer.py`
**Simply:** Learns the vocabulary from raw text.
**Imports:** `Counter`, `defaultdict` (Python math/counting tools)
**Avoids:** `Tokenizer` classes (it doesn't care about the final app)
**Why separated:** Training requires heavy math and counting. By keeping this separate, the final "Tokenizer" we use every day stays extremely fast and lightweight!

### Job 2 — Encoding
**File:** `src/abctokz/tokenizer.py`
**Simply:** Takes a string and turns it into tokens using a pipeline.
**Imports:** `Normalizer`, `PreTokenizer`, `Model`, `PostProcessor`
**Avoids:** Math algorithms or training logic.
**Why separated:** It acts like a factory manager. It just passes the text down the assembly line (normalize -> split -> tokenize) without doing the hard work itself.

### Job 3 — Saving and Loading
**File:** Split between `src/abctokz/tokenizer.py` & `src/abctokz/vocab/serialization.py`
**Simply:** Saves your trained tokenizers to folders on your computer.
**Imports:** `json`, `hashlib` (in `tokenizer.py`)
**Avoids:** Using a unified saving manager.
**Why separated:** It's *supposed* to be cleanly separated, but right now it's a mess (see the "Blurry Boundary" section below).

### Job 4 — Measuring Quality
**File:** `src/abctokz/eval/metrics.py`
**Simply:** Calculates scores like "fertility" to see if the tokenizer is good.
**Imports:** `Encoding` (the data structure it needs to check)
**Avoids:** Any actual Tokenizer code.
**Why separated:** Just like a referee shouldn't play in the game, the grading logic shouldn't be mixed into the code that actually does the tokenization.

### Job 5 — Comparing External Tokenizers
**File:** `src/abctokz/adapters/hf.py`
**Simply:** Acts as a translator so we can test HuggingFace tokenizers using our own code.
**Imports:** `Encoding`, `AdapterError`
**Avoids:** It *avoids* importing HuggingFace (`transformers`) at the top of the file!
**Why separated:** See the "Clean Boundary" section below for why this is amazing.

---

## 🌟 Big Win: A Perfectly Clean Boundary

**Where:** `src/abctokz/adapters/hf.py`

**The actual code in the file:**
```python
class HFTokenizerAdapter:
    def __init__(self, model_name_or_path: str) -> None:
        try:
            from tokenizers import Tokenizer as HFTokenizer  # <-- IMPORT IS HIDDEN DOWN HERE!
```

**Why this is great:** 
Notice how the import `from tokenizers` is pushed down *inside* the function? 
This means our core code never accidentally loads massive outside libraries unless the user specifically asks for it! 

**Simple Analogy:** 
It's like keeping a heavy power tool inside a locked box. You only have to carry the weight of the box if you actually unlock it to use the tool. If you don't use it, you never feel the weight!

---

## ⚠️ Big Mistake: A Blurry Boundary (Bug Source)

**Where:** Saving logic inside `src/abctokz/tokenizer.py`

**The Problem:** 
The file `vocab/serialization.py` exists specifically to handle writing files. But `tokenizer.py` ignores it! Instead, `tokenizer.py` manually tries to write `manifest.json` and `config.json` itself using hardcoded `json.dumps()`. 

**The Bug it Caused:**
Because `tokenizer.py` was trying to do a job it wasn't built for, it forgot to save the PreTokenizer and Normalizer settings. When you load the tokenizer later, it comes back broken! 

**Simple Analogy:**
Imagine a restaurant has a dedicated Dishwasher (serialization.py). But the Head Chef (tokenizer.py) decides to wash his own pans instead. The Chef is bad at dishwashing, so the pans come back dirty.

**How to Fix It:**
Take all the `json.dumps()` file-writing code out of `tokenizer.py` and move it into a new completely separate `ArtifactManager` tool.

---

## 📊 Summary Table

| Job | File | Clean or Blurry? |
| :--- | :--- | :--- |
| Training | `bpe_trainer.py` | ✅ Clean |
| Encoding | `tokenizer.py` | ✅ Clean |
| Saving/Loading | `tokenizer.py` | ❌ Blurry (Bug) |
| Measuring Quality | `metrics.py` | ⚠️ Mildly Blurry |
| Adapters | `hf.py` | 🌟 Extremely Clean |
