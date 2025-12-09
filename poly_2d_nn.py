import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import random
from typing import List, Tuple

# -----------------------------
# Utility functions
# -----------------------------
def clip(value, lower, upper):
    return min(upper, max(value, lower))

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
        angle += angle_steps[i]
    return points

def generate_polygon_image(img_size=100, num_vertices=12, avg_radius=40, 
                           irregularity=0.35, spikiness=0.2):
    label = random.randint(0,1)
    center = (img_size//2, img_size//2)
    vertices = generate_polygon(center, avg_radius, irregularity, spikiness, num_vertices)
    
    bg_color = 255 if label==0 else 0
    fg_color = 255 if label==1 else 0
    img = Image.new('L', (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)
    draw.polygon(vertices, fill=fg_color)
    
    return np.array(img, dtype=np.float32)/255.0, label, num_vertices

def generate_polygon_dataset(n_samples=1000, img_size=40, min_vertices=3, max_vertices=12):
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    y_color = np.zeros((n_samples, 1), dtype=np.float32)
    y_vertices = np.zeros((n_samples, 1), dtype=np.float32)
    
    for i in range(n_samples):
        num_vertices = random.randint(min_vertices, max_vertices)
        img, label, num_vertices = generate_polygon_image(img_size=img_size, num_vertices=num_vertices)
        X[i,0] = img
        y_color[i,0] = label
        y_vertices[i,0] = num_vertices
    return torch.tensor(X, dtype=torch.float32), \
           torch.tensor(y_color, dtype=torch.float32), \
           torch.tensor(y_vertices, dtype=torch.float32)

# -----------------------------
# CUDA device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Dataset
# -----------------------------
X_train, y_color, y_vertices = generate_polygon_dataset(n_samples=1000, img_size=100, min_vertices=3, max_vertices=12)
X_train, y_color, y_vertices = X_train.to(device), y_color.to(device), y_vertices.to(device)

# ---------------------------
# Multi-task CNN
# ---------------------------
class MultiTaskCNN(nn.Module):
    def __init__(self, input_channels=1, conv_channels=[16,32,64]):
        super().__init__()
        layers = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc_color = nn.Linear(in_ch, 1)
        self.fc_vertices = nn.Linear(in_ch, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        color_out = torch.sigmoid(self.fc_color(x))
        vertices_out = self.fc_vertices(x)
        return color_out, vertices_out

model = MultiTaskCNN().to(device)

# ---------------------------
# Losses & optimizer
# ---------------------------
criterion_color = nn.BCELoss()
criterion_vertices = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 50

# ---------------------------
# Training
# ---------------------------
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out_color, out_vertices = model(X_train)
    loss_color = criterion_color(out_color, y_color)
    loss_vertices = criterion_vertices(out_vertices, y_vertices)
    loss = loss_color + loss_vertices
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.3f} (Color: {loss_color.item():.3f}, Vertices: {loss_vertices.item():.3f})")

# ---------------------------
# Show a few training examples
# ---------------------------
X_vis = X_train[:9].cpu().numpy()
y_color_vis = y_color[:9].cpu().numpy()
y_vertices_vis = y_vertices[:9].cpu().numpy()

plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_vis[i,0], cmap='gray')
    plt.title(f"Color={int(y_color_vis[i,0])}\nVertices={int(y_vertices_vis[i,0])}")
    plt.axis('off')
plt.show()

# ---------------------------
# Visualize feature maps
# ---------------------------
def visualize_feature_maps(model, X_sample, conv_layers=[0,2,4]):
    x = X_sample
    for i, layer in enumerate(model.conv_layers):
        x = layer(x)
        if i in conv_layers:
            n_channels = x.shape[1]
            plt.figure(figsize=(15,3))
            for c in range(n_channels):
                plt.subplot(1, n_channels, c+1)
                fmap = x[0,c].detach().cpu().numpy()
                plt.imshow(fmap, cmap='gray')
                plt.axis('off')
                plt.title(f"Ch {c}")
            plt.suptitle(f"Feature maps after layer {i+1}")
            plt.show()

sample_img = X_train[0].unsqueeze(0)
visualize_feature_maps(model, sample_img)

# ---------------------------
# Visualize conv filters
# ---------------------------
def visualize_conv_filters(model):
    for i, layer in enumerate(model.conv_layers):
        if isinstance(layer, nn.Conv2d):
            n_filters = layer.weight.shape[0]
            plt.figure(figsize=(15,3))
            for f in range(n_filters):
                filt = layer.weight[f,0].detach().cpu().numpy()
                plt.subplot(1, n_filters, f+1)
                plt.imshow(filt, cmap='gray')
                plt.axis('off')
                plt.title(f"F {f}")
            plt.suptitle(f"Conv layer {i+1} filters")
            plt.show()

visualize_conv_filters(model)

# ---------------------------
# Predictions
# ---------------------------
model.eval()
with torch.no_grad():
    out_color, out_vertices = model(X_train[:9])
    out_color = (out_color>0.5).cpu().numpy()
    out_vertices = out_vertices.cpu().numpy()

plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_vis[i,0], cmap='gray')
    plt.title(f"Color True={int(y_color_vis[i,0])}, Pred={int(out_color[i,0])}\nVertices True={int(y_vertices_vis[i,0])}, Pred={int(out_vertices[i,0])}")
    plt.axis('off')
plt.show()
