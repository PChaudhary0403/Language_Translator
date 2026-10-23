"""Greedy autoregressive translation."""

import torch

from src.data import encode_sentence, PAD_ID, SOS_ID, EOS_ID


def translate(sentence, model, en_vocab, hi_vocab, device, max_len=25):
    model.eval()
    source_ids = encode_sentence(sentence, en_vocab, lowercase=True)
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_lengths = torch.tensor([len(source_ids)], dtype=torch.long)

    with torch.no_grad():
        hidden, cell = model.encoder(source, source_lengths)
        current_token = torch.tensor(
            [[SOS_ID]], dtype=torch.long, device=device
        )
        generated_ids = []

        for _ in range(max_len):
            logits, hidden, cell = model.decoder(current_token, hidden, cell)
            next_id = logits[:, -1, :].argmax(dim=-1)
            token_id = next_id.item()

            if token_id == EOS_ID:
                break
            if token_id not in (PAD_ID, SOS_ID):
                generated_ids.append(token_id)

            current_token = next_id.unsqueeze(1)

    id_to_word = {idx: word for word, idx in hi_vocab.items()}
    return " ".join(id_to_word.get(idx, "<UNK>") for idx in generated_ids)
