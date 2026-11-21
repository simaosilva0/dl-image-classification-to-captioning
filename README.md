# From Image Classification to Image Captioning

Course project for Deep Learning (IST, 2026-27), done in a group of 3.

We build a system that describes images in plain English, one step at a time. It starts with an MLP that classifies images, moves to a CNN, and ends with an encoder-decoder captioner (CNN encoder + RNN decoder) in the style of Show and Tell (Vinyals et al., 2015). An optional bonus part swaps the RNN for a Transformer decoder and tries pretrained models.

Every part uses the same dataset, COCO-Dominant10-64: 64×64 RGB images from MS-COCO where one object dominates the scene. Each image has a class label (10 classes: airplane, bus, cat, dog, elephant, giraffe, horse, motorcycle, train, truck) and five human-written captions.

| Split | Images | Captions |
|---|---|---|
| Train | 6,500 | 32,500 |
| Validation | 1,000 | 5,000 |
| Test | 1,000 | 5,000 |

## Project parts

| Part | Topic | Points |
|---|---|---|
| 1 | Theory: convolutions as constrained linear layers, gradients through time | 20 |
| 2 | Image classification with a linear baseline, MLPs and CNNs | 40 |
| 3 | Image captioning with an RNN decoder | 40 |
| 4 | Bonus: Transformer captioner (+10) and pretrained models (+10) | +20 |

## Repository layout

```
materials/
  project.pdf          project statement
  skeleton_code.zip    provided code: data loading, training loop, metrics
```

Our code will go next to `materials/` as we write it.

## Setup

Python 3.10 or newer.

```
unzip materials/skeleton_code.zip -x '__MACOSX/*'
cd skeleton_code
pip install -r requirements.txt
python load_data.py     # downloads the dataset (~80 MB) into data/
```

Part 2 runs fine on a laptop CPU. For Part 3 a GPU helps, and the free one on Google Colab is enough.

## Rules to keep in mind

- Parts 2 and 3 allow only `torch`, `torchvision.transforms` (resizing and augmentation only), `matplotlib`, `tqdm` and the skeleton code.
- No `numpy` in Parts 2 and 3. All computations and statistics use PyTorch tensors.
- Also banned there: scikit-learn, Lightning/Keras/HF Trainer, `torchvision.models`, `timm`, HF `transformers`, pretrained weights and pretrained tokenizers.
- No `nn.TransformerDecoderLayer` or other ready-made transformer blocks in Part 4.
- Report results as mean ± std over at least 3 seeds, and list the seeds used.
- Make every model selection decision on the validation set. Touch the test set only for final numbers.
- If we change the skeleton code, the report has to say what changed.
- Do not commit the dataset or model checkpoints (`.gitignore` already excludes `data/`, `*.pt` and `*.pth`).

## Deliverables

Deadline: Friday, December 11, 2026. Submit one zip in Fenix under the group's name with:

- The report as a PDF, written with the course LaTeX template. Limit: 8 pages, or 10 if we attempt Part 4, not counting references and appendices. It must include a paragraph on each member's contribution and the completed collaboration and AI use checklist. Each missing item costs 10 points.
- All the code needed to reproduce every table and figure, plus a README explaining how to run it.

We present and discuss the project in the first practical class of the week of December 14-18, 2026.
