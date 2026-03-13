from abctokz import AugenblickTokenizer
from abctokz.config.defaults import bpe_multilingual

config = bpe_multilingual(vocab_size=200)
corpus = ["corpus.txt"]

# Run 1
tok1 = AugenblickTokenizer.from_config(config)
tok1.train(corpus, config)

# Run 2
tok2 = AugenblickTokenizer.from_config(config)
tok2.train(corpus, config)

text = "भारत एक महान देश है"

enc1 = tok1.encode(text)
enc2 = tok2.encode(text)

print("Run1 IDs:", enc1.ids)
print("Run2 IDs:", enc2.ids)

if enc1.ids == enc2.ids:
    print("✅ Encoding is deterministic")
else:
    print("❌ Encoding differs")

# vocabulary check
vocab1 = tok1.get_vocab()
vocab2 = tok2.get_vocab()

if vocab1 == vocab2:
    print("✅ Vocabulary identical")
else:
    print("❌ Vocabulary differs")

print("Vocab size run1:", len(vocab1))
print("Vocab size run2:", len(vocab2))