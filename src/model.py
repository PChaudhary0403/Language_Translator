"""LSTM encoder-decoder model used by the translation project."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            input_vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, source, lengths):
        embedded = self.embedding(source)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, cell) = self.lstm(packed)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            output_vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, target, hidden, cell):
        embedded = self.embedding(target)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.fc(outputs)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, source_lengths, target_input):
        hidden, cell = self.encoder(source, source_lengths)
        logits, _, _ = self.decoder(target_input, hidden, cell)
        return logits
