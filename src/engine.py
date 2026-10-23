"""Training and validation loops."""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device, pad_id):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for sources, source_lengths, targets in loader:
        sources = sources.to(device)
        source_lengths = source_lengths.to(device)
        targets = targets.to(device)

        target_input = targets[:, :-1]
        target_labels = targets[:, 1:]

        optimizer.zero_grad()
        logits = model(sources, source_lengths, target_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        valid_tokens = (target_labels != pad_id).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


def evaluate_loss(model, loader, criterion, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sources, source_lengths, targets in loader:
            sources = sources.to(device)
            source_lengths = source_lengths.to(device)
            targets = targets.to(device)

            target_input = targets[:, :-1]
            target_labels = targets[:, 1:]
            logits = model(sources, source_lengths, target_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_labels.reshape(-1),
            )

            valid_tokens = (target_labels != pad_id).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)
