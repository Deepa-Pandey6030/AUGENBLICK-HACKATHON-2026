# Task 4: abctokz Architecture Mapping

This document maps the architectural responsibilities within the `abctokz` tokenizer codebase, analyzing module boundaries, identifying clean abstractions, and flagging blurry code or bugs.

## Responsibility Mapping: Who Does What?

| Job | Responsible File/Module | Why this separation exists |
| :--- | :--- | :--- |
| **1. Training a tokenizer**<br>*(learning vocab from text)* | `src/abctokz/trainers/` <br>*(e.g., `bpe_trainer.py`)* | Training algorithms (BPE, Unigram) are mathematically intense and require different data structures than just looking up tokens. Separating trainers ensures the final `Model` is lightweight and only contains what it needs to encode, not the heavy machinery needed to learn. |
| **2. Encoding new text**<br>*(using a trained tokenizer)* | `src/abctokz/tokenizer.py`<br>*(specifically the `encode()` method)* | This acts as the "orchestrator." Encoding is a pipeline (`Normalizer` → `PreTokenizer` → `Model` → `PostProcessor`). `tokenizer.py` simply wires these independent components together in the correct order so the user only has to call one function. |
| **3. Saving and loading**<br>*(to/from disk)* | Split between `src/abctokz/tokenizer.py` & `src/abctokz/vocab/serialization.py` | Saving a tokenizer requires writing a manifest, saving the vocabulary, and saving configurations. *(Note: As discussed below, this separation is currently flawed and is a source of bugs).* |
| **4. Measuring quality**<br>*(fertility, UNK rate, etc.)* | `src/abctokz/eval/metrics.py` | Evaluation logic is completely separate from tokenization. You don't want metric calculation logic polluting the fast, core `Model` classes. |
| **5. Comparing against external tokenizers** | `src/abctokz/adapters/` <br>*(e.g., `hf.py`, `sentencepiece.py`)* | These act as "translators". They import third-party libraries (which the core codebase avoids) and wrap them so they look exactly like an `abctokz` Tokenizer. This allows the evaluation scripts to run tests without knowing which underlying engine is running. |

---

## Architectural Analysis

### 🌟 Example of a CLEAN Boundary
**The `adapters/` module (e.g., `adapters/hf.py`)**

*   **Why it's satisfying:** If you look at the top of `adapters/hf.py`, it delays importing `tokenizers` and `transformers` until *inside* the `__init__` method. It acts as a perfect quarantine zone.
*   **The benefit:** This guarantees that the core `abctokz` tokenizer codebase never accidentally relies on massive HuggingFace libraries to function. If a user doesn't have HuggingFace installed, `abctokz` still works perfectly fine. The adapter only fails if the user explicitly tries to instantiate the HF adapter without the library installed.

### ⚠️ Example of a BLURRY Boundary (Bug Source)
**Saving and Loading (`tokenizer.py` vs `vocab/serialization.py`)**

*   **Why it's blurry:** `src/abctokz/vocab/serialization.py` exists specifically to handle writing JSON/text files for the tokenizer components (like `vocab.json` and `merges.txt`). However, `tokenizer.py` completely bypasses it when saving the outer tokenizer shell. `tokenizer.py` contains hardcoded `json.dumps()` calls, calculates its own SHA-256 checksums, and manually constructs file paths for `manifest.json` and `config.json`.
*   **The consequence:** Because `tokenizer.py` is trying to manage the filesystem itself, it does a poor job. This is the exact root cause of **Bug 3**! It forgot to serialize and save the Normalizer and PreTokenizer configurations because the saving logic was isolated inside the `AugenblickTokenizer.save()` method rather than a dedicated, comprehensive serialization manager.
*   **How I would fix it:** I would remove filesystem logic entirely from `tokenizer.py`. I would create a new class—something like `ArtifactManager`—inside `serialization.py` whose *sole* responsibility is taking a complete `AugenblickTokenizer` object and correctly orchestrating the writes for the config, manifest, vocab, and special tokens to a directory.
