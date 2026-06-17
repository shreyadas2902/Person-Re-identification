import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticReIDDataset(Dataset):
    def __init__(self, num_identities=40, images_per_id=6):
        self.data = []
        for pid in range(num_identities):
            # Simulate each person as a cluster in feature space
            center = torch.randn(3, 64, 32)
            for _ in range(images_per_id):
                # Add small noise to simulate lighting/pose variation
                img = center + 0.1 * torch.randn(3, 64, 32)
                self.data.append((img, pid))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.data   = base_dataset.data
        self.labels = [d[1] for d in self.data]
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.data[idx]

        pos_idx   = random.choice([i for i in self.label_to_indices[anchor_label] if i != idx])
        neg_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        neg_idx   = random.choice(self.label_to_indices[neg_label])

        return anchor_img, self.data[pos_idx][0], self.data[neg_idx][0]


# ── Model ─────────────────────────────────────────────────────────────────────

class ReIDModel(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()

        # Lightweight CNN backbone (mimics ResNet-50 concept without the size)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2))
        )

        # Channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )

        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        attn     = self.attention(features).unsqueeze(-1).unsqueeze(-1)
        features = features * attn
        return nn.functional.normalize(self.embedder(features), p=2, dim=1)


# ── Triplet Loss ──────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        return torch.clamp(pos_dist - neg_dist + self.margin, min=0).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device, epochs=8):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  |  Loss: {total_loss/len(loader):.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def get_embeddings(model, dataset, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for img, label in dataset:
            emb = model(img.unsqueeze(0).to(device)).cpu().numpy()
            embeddings.append(emb[0])
            labels.append(label)
    return np.array(embeddings), np.array(labels)


def rank1_accuracy(embeddings, labels):
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -1)
    correct = sum(labels[np.argmax(sim[i])] == labels[i] for i in range(len(labels)))
    return correct / len(labels)


# ── 3 Example Queries ─────────────────────────────────────────────────────────

def show_examples(embeddings, labels, n=3):
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -1)

    seen, count = set(), 0
    for query_idx in range(len(labels)):
        pid = labels[query_idx]
        if pid in seen or count >= n:
            continue
        seen.add(pid)
        count += 1

        top5     = np.argsort(sim[query_idx])[::-1][:5]
        matches  = labels[top5]
        scores   = sim[query_idx][top5]

        print(f"{'='*50}")
        print(f"  Query → Person ID {pid}")
        print(f"{'='*50}")
        print(f"  {'Rank':<6} {'Matched ID':<14} {'Similarity':<12} Result")
        print(f"  {'-'*42}")
        for rank, (mid, score) in enumerate(zip(matches, scores), 1):
            tag = "✓ correct" if mid == pid else "✗ wrong"
            print(f"  {rank:<6} Person {mid:<8}   {score:.4f}       {tag}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

base_dataset    = SyntheticReIDDataset(num_identities=40, images_per_id=6)
triplet_dataset = TripletDataset(base_dataset)
loader          = DataLoader(triplet_dataset, batch_size=32, shuffle=True)

model     = ReIDModel(embedding_dim=64).to(device)
criterion = TripletLoss(margin=0.3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training with Triplet Loss (ResNet-inspired CNN + attention)...")
train(model, loader, optimizer, criterion, device, epochs=8)

embeddings, labels = get_embeddings(model, base_dataset, device)
print(f"\nRank-1 Accuracy: {rank1_accuracy(embeddings, labels):.2%}")
print(f"Identities: {len(set(labels))}  |  Total images: {len(labels)}\n")

print("Example Queries — finding the same person across different images:\n")
show_examples(embeddings, labels)
