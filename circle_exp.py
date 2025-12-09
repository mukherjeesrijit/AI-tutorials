import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import csv
import matplotlib.pyplot as plt


def generate_circle_data(n_samples=1000, radius=1.0, noise_std=0.05):
    X = np.random.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.array([1 if xi**2 + yi**2 <= radius**2 else 0 for xi, yi in X])
    X += np.random.normal(0, noise_std, X.shape)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[10], output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hausdorff_from_model(model, resolution=0.01):
    xx, yy = np.meshgrid(np.arange(-1.5, 1.5, resolution),
                         np.arange(-1.5, 1.5, resolution))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        Z = model(grid).reshape(xx.shape).numpy()

    cs = plt.contour(xx, yy, Z, levels=[0.5])
    pred_boundary = cs.allsegs[0][0]

    theta = np.linspace(0, 2*np.pi, 400)
    circle_pts = np.vstack([np.cos(theta), np.sin(theta)]).T

    h1 = directed_hausdorff(circle_pts, pred_boundary)[0]
    h2 = directed_hausdorff(pred_boundary, circle_pts)[0]
    return max(h1, h2)


def train_and_eval(hidden_layers, runs=10, epochs=100):
    losses = []
    hausdorffs = []

    for _ in range(runs):
        model = MLP(hidden_layers=hidden_layers)
        X, y = generate_circle_data()

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        # compute Hausdorff distance
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300),
                             np.linspace(-1.5, 1.5, 300))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        with torch.no_grad():
            Z = model(grid).reshape(xx.shape).numpy()

        # predicted boundary
        cs = plt.contour(xx, yy, Z, levels=[0.5])
        pred_boundary = cs.allsegs[0][0]

        # true circle
        theta = np.linspace(0, 2*np.pi, 400)
        circle_pts = np.vstack([np.cos(theta), np.sin(theta)]).T

        h1 = directed_hausdorff(circle_pts, pred_boundary)[0]
        h2 = directed_hausdorff(pred_boundary, circle_pts)[0]
        hd = max(h1, h2)
        hausdorffs.append(hd)

        plt.close()  # prevent plotting

    total_params = sum(p.numel() for p in model.parameters())

    return total_params, np.mean(losses), np.mean(hausdorffs)


def run_experiments_and_save_csv():
    single_layer_sizes = list(range(1, 80, 5))
    two_layer_sizes = [1, 5, 10, 15, 20]

    rows = []

    # Single-layer experiments
    for h in single_layer_sizes:
        params, avg_loss, avg_hd = train_and_eval([h])
        rows.append(([h], params, avg_loss, avg_hd))

    # Two-layer experiments
    for h1 in two_layer_sizes:
        for h2 in two_layer_sizes:
            params, avg_loss, avg_hd = train_and_eval([h1, h2])
            rows.append(([h1, h2], params, avg_loss, avg_hd))

    # Write CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_layers", "total_params", "avg_loss", "avg_hausdorff"])
        for hidden, params, loss, hd in rows:
            writer.writerow([str(hidden), params, loss, hd])


# Run all experiments:
# run_experiments_and_save_csv()

import matplotlib.pyplot as plt
import ast
import csv

single_params = []
single_hd = []
two_params = []
two_hd = []

with open("results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hidden = ast.literal_eval(row["hidden_layers"])
        params = int(row["total_params"])
        hd = float(row["avg_hausdorff"])
        if len(hidden) == 1:
            single_params.append((params, hd))
        else:
            two_params.append((params, hd))

# Sort by total parameters
single_params = sorted(single_params, key=lambda x: x[0])
two_params = sorted(two_params, key=lambda x: x[0])

# Unpack
single_params_x, single_hd_y = zip(*single_params)
two_params_x, two_hd_y = zip(*two_params)

# Plot
plt.figure(figsize=(8,5))
plt.plot(single_params_x, single_hd_y, 'o-r', label="1 Layer")
plt.plot(two_params_x, two_hd_y, 'o-b', label="2 Layers")
plt.xlabel("Total Parameters")
plt.ylabel("Average Hausdorff Distance")
plt.title("Hausdorff Distance vs Total Parameters")
plt.legend()
plt.grid(True)
plt.show()
