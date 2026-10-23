"""Train the baseline LSTM English-to-Hindi translator.

Run from the repository root:
    python -m src.train
"""

import copy
import os
import random

import torch
import torch.nn as nn

from src.data import (
    PAD_ID, load_and_clean_pairs, split_pairs, build_vocabs, make_loaders
)
from src.engine import train_one_epoch, evaluate_loss
from src.inference import translate
from src.model import Encoder, Decoder, Seq2Seq


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 128
    hidden_dim = 256
    batch_size = 32
    max_epochs = 15
    patience = 3
    learning_rate = 0.001

    print("Device:", device)
    pairs = load_and_clean_pairs(limit=10_000, max_len=20)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    en_vocab, hi_vocab = build_vocabs(train_pairs)
    train_loader, val_loader, test_loader = make_loaders(
        train_pairs, val_pairs, test_pairs, en_vocab, hi_vocab, batch_size
    )

    print(f"Clean sentence pairs: {len(pairs)}")
    print(f"Train/validation/test: {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")
    print(f"English vocabulary: {len(en_vocab)}")
    print(f"Hindi vocabulary: {len(hi_vocab)}")

    encoder = Encoder(len(en_vocab), embedding_dim, hidden_dim, PAD_ID)
    decoder = Decoder(len(hi_vocab), embedding_dim, hidden_dim, PAD_ID)
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, PAD_ID
        )
        val_loss = evaluate_loss(
            model, val_loader, criterion, device, PAD_ID
        )
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation loss did not improve.")
            break

    model.load_state_dict(best_state)

    # Use the held-out test set only after model selection is complete.
    test_loss = evaluate_loss(model, test_loader, criterion, device, PAD_ID)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "en_vocab": en_vocab,
            "hi_vocab": hi_vocab,
            "config": {
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "pad_id": PAD_ID,
            },
            "best_validation_loss": best_val_loss,
            "test_loss": test_loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    print("\nSample test translations:")
    for en, expected_hi in test_pairs[:10]:
        predicted_hi = translate(
            en, model, en_vocab, hi_vocab, device
        )
        print(f"EN: {en}")
        print(f"Expected: {expected_hi}")
        print(f"Predicted: {predicted_hi}")
        print("-" * 50)


if __name__ == "__main__":
    main()
