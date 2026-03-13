from pathlib import Path
from abctokz.tokenizer import AugenblickTokenizer


def load_tokenizer(path):
    """
    Load trained tokenizer artifact
    """
    print(f"\nLoading tokenizer from: {path}")
    return AugenblickTokenizer.load(path)


def test_cases(tokenizer, model_name):

    inputs = [
        "hello world",
        "pseudopseudohypoparathyroidism",   # rare word
        "नमस्ते",                           # devanagari
        "🙂🚀🔥",                           # emoji
        "§¶∆",                              # rare symbols
        "𓀀",                                # ancient unicode
    ]

    print(f"\n===== Testing {model_name} =====")

    for text in inputs:

        enc = tokenizer.encode(text)

        tokens = enc.tokens

        print("\nInput:", text)
        print("Tokens:", tokens)

        if "<unk>" in tokens:
            print("⚠️  UNK produced")
        else:
            print("✓ No UNK")


def main():

    base = Path("../artifacts")

    tokenizers = {
        "WordLevel": base / "wordlevel_tok",
        "BPE": base / "bpe_tok",
        "Unigram": base / "unigram_tok"
    }

    for name, path in tokenizers.items():

        tok = load_tokenizer(path)

        test_cases(tok, name)


if __name__ == "__main__":
    main()