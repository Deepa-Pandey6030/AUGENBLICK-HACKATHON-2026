import time
from abctokz import AugenblickTokenizer
from abctokz.config.defaults import bpe_multilingual

config = bpe_multilingual(vocab_size=100)
corpus = ["corpus.txt"]

print("\n----- Determinism Experiment -----\n")

# -------------------------
# Training Run 1
# -------------------------
start1 = time.time()

tok1 = AugenblickTokenizer.from_config(config)
tok1.train(corpus, config)

end1 = time.time()
time1 = end1 - start1

# -------------------------
# Training Run 2
# -------------------------
start2 = time.time()

tok2 = AugenblickTokenizer.from_config(config)
tok2.train(corpus, config)

end2 = time.time()
time2 = end2 - start2

# -------------------------
# Encoding Test
# -------------------------
text = "भारत एक महान देश है"

enc1 = tok1.encode(text)
enc2 = tok2.encode(text)

print("Encoded IDs Run1:", enc1.ids)
print("Encoded IDs Run2:", enc2.ids)

if enc1.ids == enc2.ids:
    print("✅ Encoding output is deterministic\n")
else:
    print("❌ Encoding output differs\n")

# -------------------------
# Vocabulary Comparison
# -------------------------
vocab1 = tok1.get_vocab()
vocab2 = tok2.get_vocab()

if vocab1 == vocab2:
    print("✅ Vocabulary contents identical")
else:
    print("❌ Vocabulary differs")

print("Vocabulary Size Run1:", len(vocab1))
print("Vocabulary Size Run2:", len(vocab2))


# -------------------------
# Vocabulary Ordering Check
# -------------------------
ordered_vocab1 = sorted(vocab1.items(), key=lambda x: x[1])
ordered_vocab2 = sorted(vocab2.items(), key=lambda x: x[1])

if ordered_vocab1 == ordered_vocab2:
    print("✅ Vocabulary ordering identical (merge rules consistent)")
else:
    print("❌ Vocabulary ordering differs")


# -------------------------
# Benchmark Timing
# -------------------------
print("\nTraining Time Run1:", round(time1, 4), "seconds")
print("Training Time Run2:", round(time2, 4), "seconds")

if abs(time1 - time2) < 0.5:
    print("⚠️ Timing differs slightly due to system load (expected)")
else:
    print("⚠️ Timing difference observed — does not affect determinism")

print("\n----- Experiment Complete -----")