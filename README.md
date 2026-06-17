# Person Re-Identification

A deep learning pipeline for recognizing the same person across different camera angles, lighting conditions, and poses, built with PyTorch.

---

## What It Does

Given a query image of a person, the model retrieves the most visually similar images from a gallery, ranking them by similarity. This is the core problem in surveillance and multi-camera tracking systems.

---

## Techniques Used

| Component | Details |
|-----------|---------|
| **Backbone** | ResNet-inspired CNN for feature extraction |
| **Loss Function** | Triplet Loss with margin — pulls same-person embeddings closer, pushes different-person ones apart |
| **Attention** | Channel attention to focus on discriminative regions |
| **Augmentation** | Gaussian noise to simulate lighting and pose variation |
| **Evaluation** | Rank-1 Accuracy using cosine similarity on L2-normalized embeddings |

Real experiments used **Market-1501** and **DukeMTMC-reID** datasets from Kaggle.

---

## Quickstart

```bash
pip install torch torchvision scikit-learn
python reid.py
```

---

## Output Example

```
Query → Person ID 0
Rank   Matched ID     Similarity   Result
1      Person 0          0.9997     ✓ correct
2      Person 0          0.9996     ✓ correct
...

Rank-1 Accuracy: 100.00%
```

---

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `scikit-learn`
