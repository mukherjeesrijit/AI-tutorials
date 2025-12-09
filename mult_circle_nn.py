import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# ---------------------------
# 1. Generate training data for N circles
# ---------------------------
def generate_multi_circle_data(n_samples=1000, circle_centers=[(0,0)], radius=1.0, noise_std=0.1):
    X = np.random.uniform(-5, 5, (n_samples, 2))  # adjust square bounds as needed
    y = np.array([1 if any((xi - cx)**2 + (yi - cy)**2 <= radius**2 for cx, cy in circle_centers) else 0
                  for xi, yi in X])
    X += np.random.normal(0, noise_std, X.shape)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Example: two disconnected circles
circle_centers = [(-1, 0), (2, 0)]
radius = 1
X_train, y_train = generate_multi_circle_data(2000, circle_centers, radius, noise_std=0.1)

# ---------------------------
# 2. Define flexible MLP
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[7, 6], output_dim=1):
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

# Customize hidden layers
hidden_layers = [5000]
model = MLP(hidden_layers=hidden_layers)

# ---------------------------
# 3. Training
# ---------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# ---------------------------
# 4. General decision boundary plot
# ---------------------------
def plot_decision_boundary_general(model, X, y, circle_centers=None, radius=1.0, resolution=0.01):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        Z = model(grid).reshape(xx.shape).numpy()

    plt.contourf(xx, yy, Z, levels=[0,0.5,1], alpha=0.3, colors=['blue','orange'])
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap='bwr', edgecolor='k', s=20)

    cs = plt.contour(xx, yy, Z, levels=[0.5], colors='black')
    pred_boundary = cs.allsegs[0][0]

    # Compute Hausdorff distance against all circles
    if circle_centers is None:
        circle_centers = [(0,0)]
    all_circle_pts = []
    theta = np.linspace(0, 2*np.pi, 400)
    for cx, cy in circle_centers:
        pts = np.vstack([cx + radius*np.cos(theta), cy + radius*np.sin(theta)]).T
        all_circle_pts.append(pts)
    all_circle_pts = np.vstack(all_circle_pts)

    h1 = directed_hausdorff(all_circle_pts, pred_boundary)[0]
    h2 = directed_hausdorff(pred_boundary, all_circle_pts)[0]
    hausdorff_dist = max(h1, h2)
    print("Hausdorff distance:", hausdorff_dist)

    # Plot true circles
    for cx, cy in circle_centers:
        circle = plt.Circle((cx, cy), radius, color='green', fill=False, linewidth=2, linestyle='dashed')
        plt.gca().add_artist(circle)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')
    plt.show()

    return hausdorff_dist

# ---------------------------
# 5. Run plot and print stats
# ---------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"{hidden_layers} has total parameters:", total_params)
print("loss:", loss.item())

hausdorff_distance = plot_decision_boundary_general(model, X_train.numpy(), y_train.numpy(),
                                                    circle_centers=circle_centers, radius=radius)
