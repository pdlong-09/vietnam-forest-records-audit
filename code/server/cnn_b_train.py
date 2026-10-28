"""Paper B, the analysis plan (post hoc): ResNet18 on the VLM-B Sentinel-2 tiles, binary agriculture 2020, cross-fitted.

    python cnn_b_train.py --data cnn_b.npz --out preds_cnn.csv

cnn_b.npz: pid [n], X uint8 [n, 24, 64, 64] (4 years x true + false colour RGB), y [n] (-1 = no label), fold [n] (-1 = not
phase 2). For fold k: train on the labelled points outside fold k (20 epochs, AdamW lr 3e-4, wd 1e-4, batch 32, seed 0,
flips and 90-degree rotations), last epoch; predict with the 8 dihedral views averaged. Phase-2 points get their
out-of-fold value, the other points the mean of the 5 models. ImageNet weights: torchvision ResNet18 (local cache).
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def model24():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    w = m.conv1.weight.data  # [64, 3, 7, 7]
    conv = nn.Conv2d(24, 64, 7, 2, 3, bias=False)
    conv.weight.data = w.repeat(1, 8, 1, 1) / 8
    m.conv1 = conv
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def prep(xb, dev):
    x = torch.from_numpy(xb).to(dev).float() / 255.0
    x = (x - 0.45) / 0.25
    return F.interpolate(x, size=128, mode="bilinear", align_corners=False)


def views(x):
    for k in range(4):
        r = torch.rot90(x, k, (2, 3))
        yield r
        yield torch.flip(r, (3,))


def predict(m, X, dev, bs=128):
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x = prep(X[s:s + bs], dev)
            out.append(torch.stack([torch.sigmoid(m(v)[:, 0]) for v in views(x)]).mean(0).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    z = np.load(a.data)
    pid, X, y, fold = z["pid"], z["X"], z["y"], z["fold"]  # read each key once
    dev = "cuda"
    oof = np.full(len(pid), np.nan)
    rest = np.zeros(len(pid))
    for k in range(5):
        torch.manual_seed(0); np.random.seed(0)
        tr = np.flatnonzero((fold >= 0) & (fold != k) & (y >= 0))
        te = np.flatnonzero(fold == k)
        m = model24().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(k)
        t0 = time.time()
        for ep in range(a.epochs):
            m.train()
            idx = rng.permutation(tr)
            tot = 0.0
            for s in range(0, len(idx), 32):
                b = idx[s:s + 32]
                x = prep(X[b], dev)
                kk = int(rng.integers(4))
                x = torch.rot90(x, kk, (2, 3))
                if rng.random() < 0.5:
                    x = torch.flip(x, (3,))
                loss = F.binary_cross_entropy_with_logits(m(x)[:, 0], torch.from_numpy(y[b].astype(np.float32)).to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            print(f"fold {k} epoch {ep + 1} loss {tot / len(idx):.4f} {time.time() - t0:.0f} s", flush=True)
        oof[te] = predict(m, X[te], dev)
        rest += predict(m, X, dev) / 5
        print(f"fold {k} done", flush=True)
    f = np.where(fold >= 0, oof, rest)
    with open(a.out, "w", newline="") as fo:  # no pandas in the shared torch environment
        w = csv.writer(fo)
        w.writerow(["pid", "f_cnn", "kind"])
        w.writerows([int(p_), float(v), "out_of_fold" if k_ >= 0 else "mean_of_5"] for p_, v, k_ in zip(pid, f, fold))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
