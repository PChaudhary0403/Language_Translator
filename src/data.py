"""Dataset loading, cleaning, vocabulary, and batching utilities."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

Pair = Tuple[str, str]

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = range(4)

# Based on artifacts observed while inspecting the IIT Bombay corpus.
UI_ARTIFACT_RE = re.compile(r"\w?_\w?|\s_\s")
METADATA_ARTIFACT_RE = re.compile(
    r"#-#-#|(?:^|\s)[\w.+-]+\.po\s*\(",
    flags=re.IGNORECASE,
)


def load_and_clean_pairs(limit: int = 10_000, max_len: int = 20) -> List[Pair]:
    """Load a reproducible subset and remove obvious empty/noisy pairs.

    This is a heuristic cleaning pipeline, not a guarantee that every remaining
    translation is correct.
    """
    dataset = load_dataset(
        "cfilt/iitb-english-hindi",
        split=f"train[:{limit}]",
    )

    pairs: List[Pair] = []
    seen = set()

    for item in dataset:
        en = item["translation"]["en"].strip()
        hi = item["translation"]["hi"].strip()

        if not en or not hi:
            continue

        pair = (en, hi)
        if pair in seen:
            continue
        seen.add(pair)

        if not (1 <= len(en.split()) <= max_len):
            continue
        if not (1 <= len(hi.split()) <= max_len):
            continue

        # Remove UI accelerator fragments such as "C _ lear".
        if UI_ARTIFACT_RE.search(en) or UI_ARTIFACT_RE.search(hi):
            continue

        # Remove clear translation-file metadata from the Hindi target.
        # Keep legitimate technical words such as "GNOME" when used normally.
        if METADATA_ARTIFACT_RE.search(hi):
            continue

        pairs.append(pair)

    return pairs


def split_pairs(
    pairs: Sequence[Pair],
    random_state: int = 42,
) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Create 80/10/10 train/validation/test splits."""
    train_pairs, temp_pairs = train_test_split(
        list(pairs), test_size=0.20, random_state=random_state
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.50, random_state=random_state
    )
    return train_pairs, val_pairs, test_pairs


def build_vocab(counter: Counter) -> Dict[str, int]:
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for word, _frequency in counter.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def build_vocabs(train_pairs: Sequence[Pair]):
    """Build vocabularies from training data only to avoid data leakage."""
    en_counter = Counter()
    hi_counter = Counter()

    for en, hi in train_pairs:
        en_counter.update(en.lower().split())
        hi_counter.update(hi.split())

    return build_vocab(en_counter), build_vocab(hi_counter)


def encode_sentence(sentence: str, vocab: Dict[str, int], lowercase: bool = False):
    if lowercase:
        sentence = sentence.lower()
    token_ids = [vocab.get(word, UNK_ID) for word in sentence.split()]
    return [SOS_ID] + token_ids + [EOS_ID]


class TranslationDataset(Dataset):
    def __init__(self, pairs: Sequence[Pair], en_vocab, hi_vocab):
        self.pairs = list(pairs)
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, hi = self.pairs[idx]
        source = encode_sentence(en, self.en_vocab, lowercase=True)
        target = encode_sentence(hi, self.hi_vocab)
        return (
            torch.tensor(source, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def collate_batch(batch, pad_id: int = PAD_ID):
    sources, targets = zip(*batch)
    source_lengths = torch.tensor([len(s) for s in sources], dtype=torch.long)
    sources = pad_sequence(sources, batch_first=True, padding_value=pad_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    return sources, source_lengths, targets


def make_loaders(
    train_pairs,
    val_pairs,
    test_pairs,
    en_vocab,
    hi_vocab,
    batch_size: int = 32,
):
    from functools import partial

    collate_fn = partial(collate_batch, pad_id=PAD_ID)
    train_loader = DataLoader(
        TranslationDataset(train_pairs, en_vocab, hi_vocab),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TranslationDataset(val_pairs, en_vocab, hi_vocab),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        TranslationDataset(test_pairs, en_vocab, hi_vocab),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
