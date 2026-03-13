"""Task 1: Trace the full encode() pipeline on a Sanskrit mantra.

Stages:  Normalizer → Pre-tokenizer → BPE Model → Decoder

This script trains a BPE tokenizer and prints the output at each stage.
We use the tokenizer BEFORE save/load so normalizer + pre-tokenizer are active.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual

# --- Training corpus (multilingual) ---
CORPUS_LINES = [
    # English
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "tokenization is important for natural language processing",
    "machine learning models need good tokenizers",
    "subword segmentation helps with rare words",
    # Hindi
    "नमस्ते दुनिया",
    "यह एक परीक्षण वाक्य है",
    "हिन्दी भाषा में टोकनाइजेशन",
    "भारत एक विशाल देश है",
    "मशीन लर्निंग मॉडल के लिए टोकनाइज़र",
    # Marathi
    "नमस्कार जग",
    "मराठी भाषेत टोकनायझेशन",
    "हे एक चाचणी वाक्य आहे",
    # Sanskrit / Vedic (includes all characters from the mantra)
    "ॐ भूर्भुवः स्वः",
    "तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि",
    "धियो यो नः प्रचोदयात्",
    "ॐ नमः शिवाय",
    "सर्वे भवन्तु सुखिनः",
    "ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥",
    "श्रीमद्भगवद्गीता अध्याय: ॥",
    "ॐ शान्तिः शान्तिः शान्तिः ॥",
    # Mixed
    "hello नमस्ते world दुनिया",
    "BPE tokenizer for Hindi हिन्दी",
    "Devanagari script नागरी लिपि",
] * 30

MANTRA = "ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥"


def main() -> None:
    # ---- Step 0: Train the BPE tokenizer ----
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        config = bpe_multilingual(vocab_size=500)
        tokenizer = Tokenizer.from_config(config)
        print("=" * 70)
        print("TRAINING BPE TOKENIZER")
        print("=" * 70)
        tokenizer.train([str(corpus_path)], config)
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print()

        # ---- The input ----
        print("=" * 70)
        print("INPUT")
        print("=" * 70)
        print(f"  Raw text: {MANTRA!r}")
        print()

        # ---- Stage 1: Normalizer ----
        print("=" * 70)
        print("STAGE 1 — NORMALIZER  (DevanagariNormalizer)")
        print("  File: src/abctokz/normalizers/devanagari.py")
        print("=" * 70)
        normalized = tokenizer._normalizer.normalize(MANTRA)
        print(f"  Input:      {MANTRA!r}")
        print(f"  Normalized: {normalized!r}")
        print(f"  Changed?    {MANTRA != normalized}")
        if MANTRA != normalized:
            for i, (a, b) in enumerate(zip(MANTRA, normalized)):
                if a != b:
                    print(f"    Diff at pos {i}: {a!r} (U+{ord(a):04X}) → {b!r} (U+{ord(b):04X})")
        print()

        # ---- Stage 2: Pre-tokenizer ----
        print("=" * 70)
        print("STAGE 2 — PRE-TOKENIZER  (DevanagariAwarePreTokenizer)")
        print("  File: src/abctokz/pretokenizers/devanagari_aware.py")
        print("=" * 70)
        pre_tokens = tokenizer._pretokenizer.pre_tokenize(normalized)
        print(f"  Input:      {normalized!r}")
        print(f"  Pre-tokens ({len(pre_tokens)} pieces):")
        for i, pt in enumerate(pre_tokens):
            print(f"    [{i}] {pt!r}")
        print()

        # ---- Stage 3: BPE Model ----
        print("=" * 70)
        print("STAGE 3 — BPE MODEL  (BPEModel)")
        print("  File: src/abctokz/models/bpe.py")
        print("=" * 70)
        all_ids = []
        all_tokens = []
        for pt in pre_tokens:
            pairs = tokenizer._model.tokenize(pt)
            toks = [t for t, _ in pairs]
            ids = [i for _, i in pairs]
            all_ids.extend(ids)
            all_tokens.extend(toks)
            print(f"  Pre-token: {pt!r}")
            print(f"    → Subwords: {toks}")
            print(f"    → IDs:      {ids}")
        print()
        print(f"  FINAL token list ({len(all_tokens)} tokens): {all_tokens}")
        print(f"  FINAL ID list    ({len(all_ids)} IDs):     {all_ids}")
        print()

        # ---- Full encode() call for comparison ----
        print("=" * 70)
        print("FULL encode() OUTPUT")
        print("=" * 70)
        enc = tokenizer.encode(MANTRA)
        print(f"  tokens:  {enc.tokens}")
        print(f"  ids:     {enc.ids}")
        print(f"  offsets: {enc.offsets}")
        print(f"  length:  {len(enc.ids)} tokens")
        print()

        # ---- Stage 4: Decoder ----
        print("=" * 70)
        print("STAGE 4 — DECODER  (SubwordDecoder)")
        print("  File: src/abctokz/decoders/subword_decoder.py")
        print("=" * 70)
        decoded = tokenizer.decode(enc.ids)
        print(f"  Input IDs:  {enc.ids}")
        print(f"  Decoded:    {decoded!r}")
        print(f"  Original:   {MANTRA!r}")
        print(f"  Round-trip match: {decoded == MANTRA}")
        print()


if __name__ == "__main__":
    main()
