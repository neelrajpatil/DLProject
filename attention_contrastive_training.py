import argparse
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        diag = torch.eye(batch_size, device=device)
        mask = mask - diag

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim=1536):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        w = self.attention(x)
        alpha = torch.softmax(w, dim=0)
        pooled = (alpha * x).sum(dim=0)
        return pooled

class PatchDataset(Dataset):
    def __init__(self, features_h5, label_csv):
        import pandas as pd
        self.df = pd.read_csv(label_csv)
        with h5py.File(features_h5, 'r') as f:
            self.coords = f['coords'][:]
            self.features = f['features'][:]

        coord2idx = {}
        for i, (cx, cy) in enumerate(self.coords):
            coord2idx[(cx, cy)] = i

        self.patch_groups = defaultdict(list)
        self.labels = {}
        for _, row in self.df.iterrows():
            c = (row['coord_x'], row['coord_y'])
            if c in coord2idx:
                idx = coord2idx[c]
                patch_id = row['PATCH_ID']
                self.patch_groups[patch_id].append(idx)
                self.labels[patch_id] = int(row['GROUND_TRUTH_NUM'])

        self.patch_ids = list(self.patch_groups.keys())

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        indices = self.patch_groups[patch_id]
        x = self.features[indices]
        y = self.labels[patch_id]
        return x, y, patch_id

def collate_fn(batch):
    xs = []
    ys = []
    pids = []
    lengths = []

    for (x, y, pid) in batch:
        xs.append(torch.from_numpy(x).float())
        ys.append(y)
        pids.append(pid)
        lengths.append(x.shape[0])

    return xs, torch.tensor(ys), pids, lengths

def train_model(features_h5, label_csv, output_pt, epochs=10, lr=1e-3, temperature=0.07, batch_size=16):
    dataset = PatchDataset(features_h5, label_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = AttentionPooling(embed_dim=1536)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SupConLoss(temperature=temperature)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for xs, ys, pids, lengths in loader:
            optimizer.zero_grad()
            pooled_feats = []
            all_ys = []
            for i, x_ in enumerate(xs):
                x_ = x_.cuda()
                pooled = model(x_)
                pooled_feats.append(pooled.unsqueeze(0))
                all_ys.append(ys[i].item())

            pooled_feats = torch.cat(pooled_feats, dim=0)
            all_ys = torch.tensor(all_ys).long().cuda()

            loss = criterion(pooled_feats, all_ys)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    all_patch_ids = []
    all_pooled = []
    all_labels = []
    with torch.no_grad():
        for xs, ys, pids, lengths in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn):
            for i, x_ in enumerate(xs):
                x_ = x_.cuda()
                pooled = model(x_)
                all_patch_ids.append(pids[i])
                all_labels.append(ys[i].item())
                all_pooled.append(pooled.cpu().numpy())

    all_pooled = np.vstack(all_pooled)
    all_labels = np.array(all_labels)
    np.savez(output_pt, embeddings=all_pooled, labels=all_labels, patch_ids=all_patch_ids)
    print(f"Refined patch-level embeddings saved to {output_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform attention pooling and supervised contrastive learning on extracted embeddings.")
    parser.add_argument("--features_h5", type=str, required=True, help="Path to the .h5 file with extracted sub-patch embeddings")
    parser.add_argument("--label_csv", type=str, required=True, help="CSV file with (slide_id, coord_x, coord_y, PATCH_ID, GROUND_TRUTH_NUM)")
    parser.add_argument("--output_pt", type=str, required=True, help="Output .npz file to store refined embeddings")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_model(args.features_h5, args.label_csv, args.output_pt, epochs=args.epochs, lr=args.lr, temperature=args.temperature, batch_size=args.batch_size)
