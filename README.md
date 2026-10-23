# LSTM English–Hindi Translator

A learning project implementing an English-to-Hindi sequence-to-sequence translator using a PyTorch LSTM encoder and decoder.

> This is an educational baseline, not a production-quality translator. It was trained on a small subset of the IIT Bombay English–Hindi parallel corpus, which is heavily represented by technical/software UI text. Translation quality and generalization are limited.

## What this project demonstrates

- Loading and cleaning parallel sentence pairs
- Train/validation/test splits
- Word-level vocabularies with `<PAD>`, `<SOS>`, `<EOS>`, and `<UNK>`
- Embeddings and LSTM encoder–decoder states
- Padded batches and packed source sequences
- Teacher forcing, cross-entropy loss, gradient clipping, and Adam
- Validation-based early stopping and held-out test evaluation
- Greedy autoregressive decoding

## Repository structure

```text
translator-project/
├── notebooks/
│   └── Translator.ipynb   # Original Colab notebook, preserved
├── src/
│   ├── data.py            # Dataset, cleaning, vocabulary, batching
│   ├── model.py           # Encoder, decoder, Seq2Seq
│   ├── engine.py          # Training and evaluation loops
│   ├── inference.py       # Greedy decoding
│   └── train.py           # End-to-end training entry point
├── checkpoints/           # Generated locally; ignored by Git
├── requirements.txt
├── .gitignore
└── README.md
```

The original notebook is retained so its cell-by-cell history and outputs are not lost. The `src/` modules are a cleaner, reusable version of the core pipeline.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train

From the repository root:

```bash
python -m src.train
```

The script downloads the first 10,000 examples from `cfilt/iitb-english-hindi`, applies heuristic cleaning, creates an 80/10/10 split, builds vocabularies from the training split only, trains with early stopping, evaluates on the held-out test set, and saves `checkpoints/best_model.pt`.

The first run requires internet access to download the dataset. CUDA is used when available; otherwise training runs on CPU.

## Data cleaning notes

The cleaning rules remove empty pairs, duplicate pairs, sentences longer than 20 whitespace-separated tokens, obvious UI accelerator artifacts such as `C _ lear`, and clear translation-file metadata signatures in the Hindi target.

These rules are heuristic. They do not guarantee that all remaining translations are accurate, and they may leave noisy or awkward reference translations. Inspect samples before making claims about quality.

## Limitations

- Small training subset and narrow technical/software domain
- Simple whitespace tokenization and word-level vocabulary
- No attention mechanism
- Greedy decoding
- No formal BLEU/chrF evaluation in this baseline
- Dataset reference translations can be incomplete, inconsistent, or noisy

## Reproducibility

The split uses `random_state=42`, and training seeds PyTorch and Python's `random` module. Results can still vary across hardware and library versions.

## Dataset attribution

Dataset: [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) on Hugging Face. Review the dataset card and applicable terms before redistributing dataset contents.
