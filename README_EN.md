# New Intent Discovery for Educational Dialogue Texts via Semantic-Aware Data Augmentation

<div align="center">

**[中文](README.md) | [English](README_EN.md)**

</div>

---

## Project Abstract

Understanding educational intents in teacher-student dialogues is crucial for identifying effective teaching patterns and optimizing teaching strategies. Traditional analysis methods rely on static, pre-defined coding schemes, but these schemes struggle to adapt to diverse educational scenarios, and manually optimizing to capture new educational intents requires expensive expert costs.

Although data-driven techniques are significant for automatically discovering new educational intents from large-scale unlabeled corpora, research in this direction remains relatively insufficient.

Therefore, this project proposes the **New Educational Intent Discovery (NEID)** method. We design a novel transfer learning framework that extends the two-stage New Intent Discovery (NID) method to NEID and enhances semantic-aware discriminative representation capabilities:

## Project Architecture
1. In stage 1, we perform Multi-Task Pre-training on the public and task datasets.
2. In stage 2, we perform Knowledge Distillation, where the student model internalizes noise detection capabilities under the guidance of the teacher
model.
3. In stage 3, we perform SADA-Empowered Contrastive Learning with nearest neighbors. After training, we employ a simple non-parametric clustering
algorithm to obtain clustering results, where each cluster indicates a specific
type of educational intent.

### Project Structure

```
.
├── main.py                    # Main program entry
├── sada.py                    # Contrastive learning and nearest neighbors
├── mtp.py                     # Multi-task pre-training
├── intent_pretrain.py         # Intent pre-training manager
├── methods.py                 # Knowledge distillation and SADA methods
├── model.py                   # Model architecture definition
├── dataloader.py              # Data loading and processing
├── init_parameter.py          # Parameter initialization
├── requirements.txt           # Python dependencies
│
├── scripts/
│   └── SADA_talkmoves.sh      # Running script
│
├── utils/
│   ├── tools.py               # Utility functions
│   ├── memory.py              # Memory bank implementation
│   ├── contrastive.py         # Contrastive learning components
│   ├── build_ml.py            # Model building utilities
│   ├── adamW.py               # AdamW optimizer
│   ├── neighbor_dataset.py    # Neighbor dataset
│   └── sequence_classification.py  # Sequence classification
│
├── data/                      # Data directory
│   └── talkmoves/             # TalkMoves dataset
│       ├── train.tsv          # Training set
│       ├── dev.tsv            # Validation set
│       └── test.tsv           # Test set
│
├── data_external/             # External datasets
│   └── clinc/                 # CLINC150 dataset
│       └── dataset.json       # Data file
│
```

---

## Project Implementation

### Install Dependencies

```bash
pip install -r requirements.txt
```
### Create Model Directory

**Important**: Create a `model` folder to store the downloaded pre-trained model:

```bash
mkdir -p model/bert-base-uncased
```

### Running Command

```bash
bash SADA_talkmoves.sh 0,1,2,3
```
Where `0,1,2,3` specifies the GPU IDs to use (supports multi-GPU).

### Supported Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `CUDA_VISIBLE_DEVICES` | GPU device IDs | `0,1,2,3` |
| `internal_dataset` | Internal dataset | `talkmoves` |
| `labeled_ratio` | Labeled data ratio | `0.1`, `0.25` |
| `known_cls_ratio` | Known class ratio | `0.5`, `0.75` |
| `input_strategy` | Input strategy | `CONTEXT`, `ORIGINAL` |
| `with_speaker` | Whether to use speaker info | `0`, `1` |
| `view_strategy` | View strategy | `SADA`, `rtr`, `shuffle`, `none` |

### Output Results

Running results will be saved to:

- **Model Checkpoints**: `save_models/checkpoints/`
- **Training Logs**: `logs/`
- **Visualizations**: `save_models/` (UMAP visualization)
- **Evaluation Results**: `results.csv`

---


This project references and adapts code from the following open-source projects:

1. https://github.com/zhang-yu-wei/MTP-SADA
2. https://github.com/WenxiongLiao/mask-bert