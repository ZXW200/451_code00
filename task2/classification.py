from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from utils import fmt_time, save_js


# Neural network classifier with two hidden layers
class LinClf(nn.Module):
    def __init__(self, in_dim, n_cls, h_dim=512):
        super().__init__()

        # Multi-layer perceptron with dropout for regularization
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h_dim // 2, n_cls)
        )

    # Forward pass through network
    def forward(self, x):
        return self.net(x)


# Train linear classifier with early stopping
def train_lin(
        X, y, n_cls, dev,
        te_size=0.2, bs=64, lr=0.001, eps=100,
        h_dim=512, pat=10
):
    print(f"Train Lin (dim={X.shape[1]}, cls={n_cls})")

    # Split data into train and validation sets
    xt, xv, yt, yv = train_test_split(
        X, y, test_size=te_size, random_state=42, stratify=y
    )

    # Create PyTorch datasets and data loaders
    ds_t = TensorDataset(torch.FloatTensor(xt).to(dev), torch.LongTensor(yt).to(dev))
    ds_v = TensorDataset(torch.FloatTensor(xv).to(dev), torch.LongTensor(yv).to(dev))

    dl_t = DataLoader(ds_t, batch_size=bs, shuffle=True)
    dl_v = DataLoader(ds_v, batch_size=bs, shuffle=False)

    # Initialize model, loss function and optimizer
    model = LinClf(X.shape[1], n_cls, h_dim).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    # Track training history
    log = {'t_loss': [], 't_acc': [], 'v_loss': [], 'v_acc': []}
    best_loss = float('inf')
    cnt = 0

    t0 = time.time()

    # Training loop with early stopping
    for ep in range(eps):
        # Training phase
        model.train()
        sum_loss = 0.0
        corr = 0
        tot = 0

        for xb, yb in dl_t:
            opt.zero_grad()
            out = model(xb)
            l = loss_fn(out, yb)
            l.backward()
            opt.step()

            # Accumulate loss and accuracy
            sum_loss += l.item()
            _, pred = out.max(1)
            tot += yb.size(0)
            corr += pred.eq(yb).sum().item()

        t_loss = sum_loss / len(dl_t)
        t_acc = corr / tot

        # Validation phase
        model.eval()
        v_sum = 0.0
        v_corr = 0
        v_tot = 0

        with torch.no_grad():
            for xb, yb in dl_v:
                out = model(xb)
                l = loss_fn(out, yb)
                v_sum += l.item()
                _, pred = out.max(1)
                v_tot += yb.size(0)
                v_corr += pred.eq(yb).sum().item()

        v_loss = v_sum / len(dl_v)
        v_acc = v_corr / v_tot

        # Log metrics
        log['t_loss'].append(t_loss)
        log['t_acc'].append(t_acc)
        log['v_loss'].append(v_loss)
        log['v_acc'].append(v_acc)

        # Print progress every 5 epochs
        if (ep + 1) % 5 == 0:
            print(f"Ep {ep + 1}/{eps} - L: {t_loss:.3f}/{v_loss:.3f}, A: {t_acc:.3f}/{v_acc:.3f}")

        # Early stopping logic
        if v_loss < best_loss:
            best_loss = v_loss
            cnt = 0
            best_w = model.state_dict().copy()
        else:
            cnt += 1

        if cnt >= pat:
            print(f"Early stop {ep + 1}")
            break

    # Load best model weights
    model.load_state_dict(best_w)

    dt = time.time() - t0
    print(f"Done: {fmt_time(dt)}")

    return model, log


# Train KNN classifier on features
def train_knn(X, y, te_size=0.2, k=5):
    print(f"Train KNN (k={k})")
    t0 = time.time()

    # Split data for training
    xt, xv, yt, yv = train_test_split(
        X, y, test_size=te_size, random_state=42, stratify=y
    )

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xt, yt)

    dt = time.time() - t0
    print(f"Done: {fmt_time(dt)}")

    return knn, dt


# Evaluate classifier performance on test set
def eval_clf(model, X, y, names, kind='sklearn', dev=None):
    print("Eval...")
    t0 = time.time()

    # Get predictions based on model type
    if kind == 'torch':
        model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(dev)
            out = model(xt)
            _, pred = out.max(1)
            pred = pred.cpu().numpy()
    else:
        pred = model.predict(X)

    dt = time.time() - t0

    # Calculate classification metrics
    acc = accuracy_score(y, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average='macro')
    cm = confusion_matrix(y, pred)

    res = {
        'acc': acc, 'prec': pr, 'rec': rc, 'f1': f1,
        'cm': cm.tolist(), 'time': dt, 'pred': pred.tolist()
    }

    print(f"Acc: {acc:.4f}, Prec: {pr:.4f}, Rec: {rc:.4f}, F1: {f1:.4f}")
    return res


# Plot training and validation curves
def plot_curves(log, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(log['t_loss']) + 1)

    # Plot loss curves
    ax1.plot(ep, log['t_loss'], 'b-', label='Train')
    ax1.plot(ep, log['v_loss'], 'r-', label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy curves
    ax2.plot(ep, log['t_acc'], 'b-', label='Train')
    ax2.plot(ep, log['v_acc'], 'r-', label='Val')
    ax2.set_title('Acc')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# Run classification experiments with multiple methods
def run_class(feats, y, names, dev, out, methods=['linear', 'knn']):
    n_c = len(names)
    res = {}

    # Split data into train and test sets
    xt, xv, yt, yv = train_test_split(
        feats, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Tr: {len(xt)}, Te: {len(xv)}")

    # Train and evaluate linear classifier
    if 'linear' in methods:
        print("\n1. Linear")
        mod, log = train_lin(xt, yt, n_c, dev)
        met = eval_clf(mod, xv, yv, names, 'torch', dev)
        plot_curves(log, out / 'figures' / 'lin_curve.png')
        res['linear'] = {'metrics': met, 'history': log}

    # Train and evaluate KNN classifier
    if 'knn' in methods:
        print("\n2. KNN")
        knn, dt = train_knn(xt, yt)
        met = eval_clf(knn, xv, yv, names, 'sklearn')
        res['knn'] = {'metrics': met, 'time': dt}

    # Compare methods with bar charts
    print("\nComparison")
    comp_m = ['acc', 'prec', 'rec', 'f1']
    data = {}

    # Extract metrics for comparison
    for m in methods:
        if m in res:
            data[m] = {k: res[m]['metrics'][k] for k in comp_m}

    # Create comparison plots
    fig, axes = plt.subplots(1, len(comp_m), figsize=(18, 5))

    for i, m_name in enumerate(comp_m):
        nms = list(data.keys())
        vals = [data[m][m_name] for m in nms]

        # Plot bars with value labels
        bars = axes[i].bar(nms, vals, alpha=0.7)
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{v:.3f}', ha='center', va='bottom')

        axes[i].set_title(m_name.title())
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out / 'figures' / 'comp.png', dpi=300, bbox_inches='tight')
    plt.close()

    save_js(res, out / 'results', 'class_res.json')
    return res

