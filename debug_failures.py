from abctokz.config.schemas import TokenizerConfig
from pydantic import ValidationError

print("---- Test 1: Model–Trainer mismatch ----")

try:
    config = {
        "model": {"type": "bpe", "vocab_size": 1000},
        "trainer": {"type": "wordlevel", "vocab_size": 1000}
    }

    TokenizerConfig(**config)

except Exception as e:
    print("Error detected:")
    print(e)


print("\n---- Test 2: Invalid vocab_size ----")

try:
    config = {
        "model": {"type": "bpe", "vocab_size": -10},
        "trainer": {"type": "bpe", "vocab_size": -10}
    }

    TokenizerConfig(**config)

except ValidationError as e:
    print("Validation error detected:")
    print(e)