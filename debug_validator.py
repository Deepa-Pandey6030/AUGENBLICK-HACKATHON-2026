from abctokz.config.schemas import TokenizerConfig

config = {
    "model": {"type": "bpe", "vocab_size": 1000},
    "trainer": {"type": "wordlevel", "vocab_size": 1000}
}

TokenizerConfig(**config)