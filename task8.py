#testing normalizer result 
from abctokz.normalizers.devanagari import DevanagariNormalizer

norm = DevanagariNormalizer()

sindhi = "आयो लाल, सभई चायो, झूलेलाल!"
marathi = "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"

print("Sindhi Raw:", sindhi)
print("Sindhi Normalized:", norm.normalize(sindhi))

print()

print("Marathi Raw:", marathi)
print("Marathi Normalized:", norm.normalize(marathi))

#pretokenization test

from abctokz.pretokenizers.whitespace import WhitespacePreTokenizer
from abctokz.pretokenizers.punctuation import PunctuationPreTokenizer

# Create pretokenizers
whitespace_pt = WhitespacePreTokenizer()
punct_pt = PunctuationPreTokenizer()

sindhi = "आयो लाल, सभई चायो, झूलेलाल!"
marathi = "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"

# Step 1 — whitespace split
sindhi_tokens = whitespace_pt.pre_tokenize(sindhi)
marathi_tokens = whitespace_pt.pre_tokenize(marathi)

# Step 2 — punctuation split
sindhi_final = []
for tok in sindhi_tokens:
    sindhi_final.extend(punct_pt.pre_tokenize(tok))

marathi_final = []
for tok in marathi_tokens:
    marathi_final.extend(punct_pt.pre_tokenize(tok))

print("Sindhi Pretokens:", sindhi_final)
print("Marathi Pretokens:", marathi_final)