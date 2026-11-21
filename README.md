# From Image Classification to Image Captioning

Course project for Deep Learning (IST, 2026-27), done in a group of 3.

We build a system that describes images in plain English, one step at a time. It starts with a linear classifier and an MLP, moves to a CNN, and ends with an encoder-decoder captioner (CNN encoder + GRU/LSTM decoder) in the style of Show and Tell (Vinyals et al., 2015). The optional bonus replaces the RNN with a Transformer decoder and tries pretrained models. Everything in Parts 2 and 3 is written from scratch in plain PyTorch.

## Dataset

COCO-Dominant10-64 is a subset of MS-COCO with only images where one object category dominates the scene. Each example has:

- a 64×64 RGB image (12,288 pixel values)
- a class label out of 10: airplane, bus, cat, dog, elephant, giraffe, horse, motorcycle, train, truck
- five human-written captions in English

| Split | Images | Per class | Captions |
|---|---|---|---|
| Train | 6,500 | 650 | 32,500 |
| Validation | 1,000 | 100 | 5,000 |
| Test | 1,000 | 100 | 5,000 |

Captions average about 11 tokens and 95% have at most 15. They describe the whole scene, but 94% of the training captions name the class or a synonym ("plane" for airplane). The images are small and cluttered, and the captions keep the annotators' typos.

`load_data.py` downloads the data from Hugging Face (about 80 MB, pinned to a fixed revision) into `data/` on first use and caches it as a single `.pt` file. Some Flickr image licenses are NonCommercial, so the data is for course use only.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python load_data.py      # downloads the dataset and prints the size of each split
```

Part 2 runs fine on a laptop CPU. Part 3 is faster on a GPU, and the free one on Google Colab is enough (*Runtime > Change runtime type*). On Colab, clone the repo and run the same commands with `!` in front.

## Repository layout

```
load_data.py        skeleton: download + load the dataset as tensors
data.py             skeleton: datasets and DataLoaders for classification and captioning
train.py            skeleton: generic training loop, set_seed, count_parameters, get_device
metrics.py          skeleton: accuracy, confusion matrix, BLEU-4, class mention, diversity
requirements.txt    torch, torchvision, matplotlib, tqdm, pyarrow, pillow
materials/
  project.pdf           project statement
  skeleton_code.zip     original skeleton code as handed out
  skeleton_README.md    the skeleton's own README, with a full usage example
```

Our own code (models, tokenizer, experiments, plots) goes next to the skeleton files.

## Skeleton code at a glance

| Function | What it does |
|---|---|
| `load_coco_dominant10()` | `{split: {"images", "labels", "captions", "metadata"}}`. Images are uint8 `(N, 3, 64, 64)`, labels long `(N,)`, captions a list of 5 strings per image. `CLASSES` has the class names. |
| `channel_stats(images)` | Per-channel mean and std. Compute on the training set only. |
| `image_loaders(data, batch_size, mean, std, flip)` | Train/val/test loaders of `(images, labels)`. Only train is shuffled; `flip=True` adds random horizontal flips to train. |
| `caption_loaders(data, encode, pad_id, batch_size, ...)` | Loaders of `(images, captions)`, one item per (image, caption) pair, captions right-padded with `pad_id` to the longest in the batch. We write `encode`. |
| `train(model, optimizer, loss_fn, train_loader, val_score_fn, epochs, ...)` | Training loop with optional scheduler (stepped per epoch), `patience` for early stopping and `grad_clip`. Restores the best validation epoch and returns a history dict. `val_score_fn` must return a score where higher is better. |
| `set_seed(seed)` | Seeds Python and PyTorch, which also fixes the shuffling order. |
| `accuracy`, `confusion_matrix`, `per_class_accuracy` | Classification metrics on tensors of predictions and labels. |
| `bleu4(hyps, refs)` | Corpus BLEU-4 (0 to 100), same as nltk's `corpus_bleu` without smoothing. |
| `class_mention_accuracy(hyps, labels)` | Fraction of captions that name the image's class, using `CLASS_SYNONYMS` (can be extended). |
| `distinct_ngrams(hyps, n)` | Number of distinct n-grams across all generated captions. |

The captioning metrics take lists of tokens, so tokenize references with our tokenizer and strip `<bos>`, `<eos>` and `<pad>` from generated captions first. To generate captions, iterate over `image_loaders` (one item per image, in dataset order) and compare with `data[split]["captions"]`.

Minimal classification run:

```python
import torch
import torch.nn.functional as F
from load_data import load_coco_dominant10
from data import channel_stats, image_loaders
from metrics import accuracy
from train import set_seed, train

data = load_coco_dominant10()
mean, std = channel_stats(data["train"]["images"])
train_loader, val_loader, test_loader = image_loaders(data, batch_size=64, mean=mean, std=std)

set_seed(0)
model = ...  # (B, 3, 64, 64) -> (B, 10)
history = train(model, torch.optim.SGD(model.parameters(), lr=0.01),
                loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]),
                train_loader=train_loader,
                val_score_fn=lambda m: accuracy(predict(m, val_loader), data["validation"]["labels"]),
                epochs=20)
```

`materials/skeleton_README.md` has the full example, including `predict`.

## What each part asks for

### Part 1: Theory (20 pts)

- **Q1.1** Convolutions as constrained linear layers: circulant matrix form, translation equivariance and invariance after global average pooling, parameter counts for a dense layer vs. a 3×3 conv on 64×64×3 input.
- **Q1.2** Gradients through time: Jacobian of a vanilla RNN, vanishing/exploding bound via σ_max(W), why clipping helps only with exploding gradients, and why the LSTM cell state and a positive forget-gate bias help.

### Part 2: Image classification (40 pts)

Images are flat vectors except in Q2.5.

- **Q2.1** Split sizes, class distribution, normalization stats. Linear classifier (logistic regression) with SGD: training loss and validation accuracy curves, test accuracy at the best validation epoch.
- **Q2.2** Configurable MLP (depth, width, activation, dropout). Tune on validation, report the search, curves and test accuracy (mean ± std, 3 seeds).
- **Q2.3** Ablation of at least 3 design choices, one factor at a time, same budget for all runs. Tables with mean ± std, discussion of significance and over/underfitting, per-class accuracy and confusion matrix.
- **Q2.4** Visualize first-layer weights and linear class templates. Evaluate on test images shifted by 1, 2, ... pixels, and retrain on images with a fixed random pixel permutation.
- **Q2.5** Small CNN with no more parameters than the best MLP. Compare linear vs. MLP vs. CNN and repeat the shift and permutation experiments.

### Part 3: Image captioning (40 pts)

- **Q3.1** Own tokenizer and vocabulary from the training captions, with `<unk>` (frequency threshold), `<bos>`, `<eos>`, `<pad>`. Report vocabulary size, `<unk>` rate on validation and caption length distribution, and explain how padding is ignored in the loss.
- **Q3.2** Encoder (MLP or CNN) + GRU/LSTM decoder with embeddings, trained with teacher forcing and token-level cross-entropy. Greedy decoding, loss and perplexity curves, BLEU-4 on val and test, at least 8 example captions (good and bad).
- **Q3.3** (a) MLP vs. CNN encoder, and frozen pretrained-by-us vs. jointly trained encoder, with metrics and parameter counts. (b) Class mention accuracy per encoder, compared with classification accuracy and BLEU-4.
- **Q3.4** (a) Beam search with beam sizes 1, 3, 5, with and without length normalization: BLEU-4, average length, distinct unigrams and bigrams. (b) Small example where greedy decoding misses the most probable caption, and why exact maximization favors short captions.

### Part 4: Bonus (up to +20, total capped at 100)

- **Q4.1** Transformer decoder from scratch with causal self-attention plus cross-attention to CNN features (or an image prefix). `nn.MultiheadAttention` is fine, but we write the blocks ourselves (no `nn.TransformerDecoderLayer`). Compare with the best RNN; visualize attention if using cross-attention.
- **Q4.2** Pretrained encoder (ResNet, ViT, CLIP) and/or decoder (GPT-2 with a learned prefix, as in ClipCap). Compare with our from-scratch models, including compute cost. BLIP and similar can be a reference point but not our system.

## Rules

Allowed: `torch` (including `nn`, `nn.functional`, `optim`, `utils.data`), `torchvision.transforms` for resizing and augmentation only, `matplotlib` for plots only, `tqdm`, and the skeleton code.

Not allowed in Parts 2 and 3:

- `numpy`. All computations, statistics and analyses use PyTorch tensors.
- scikit-learn, PyTorch Lightning, Keras, the Hugging Face Trainer
- `torchvision.models`, `timm`, Hugging Face `transformers`
- pretrained weights and pretrained tokenizers

`pyarrow` and `Pillow` are only used inside `load_data.py`. Part 4 relaxes some of these restrictions.

We may modify the skeleton code, but the report has to say what changed. The skeleton was committed unmodified in `30756da`, so `git diff 30756da -- load_data.py data.py train.py metrics.py` lists every change.

## Methodology

- Report results as mean ± std over at least 3 seeds and list the seeds.
- Train on train, make every model selection decision (hyperparameters, early stopping, architecture) on validation, and use test only for final numbers.
- Every number and figure in the report must be reproducible with the code in this repo.
- Grading values correct derivations, sound experiments and careful analysis over state-of-the-art numbers.

## Deliverables

Deadline: **Friday, December 11, 2026.** One zip in Fenix under the group's name containing:

- The report as a PDF, written with the course LaTeX template (Overleaf link in `materials/project.pdf`; keep its font size, margins and spacing). Maximum 8 pages, or 10 with Part 4, not counting references and appendices. Answer the questions in order using their numbers (Q2.3, ...).
- A paragraph stating each member's contribution, and the completed collaboration and AI use checklist from the end of the statement. Each one missing costs 10 points.
- All the code needed to reproduce every table and figure, with a README explaining how to run it. No datasets or model checkpoints.

We discuss the project with the instructors in the first practical class of the week of December 14-18, 2026. Every member must be able to explain any part of the submission, and individual grades can change if someone can't.
