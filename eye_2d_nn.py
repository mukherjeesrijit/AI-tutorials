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
    # normalize so that sum is 2*pi
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

# -----------------------------
# Generate a filled polygon image
# -----------------------------
def generate_polygon_image(img_size=100, num_vertices=12, avg_radius=40, 
                           irregularity=0.35, spikiness=0.2):
    """
    Generate a single image with a filled polygon.
    Returns: normalized image (0-1) and label (1=white polygon, 0=black polygon)
    """
    label = random.randint(0,1)  # 1 = white polygon, 0 = black polygon
    center = (img_size // 2, img_size // 2)
    vertices = generate_polygon(center, avg_radius, irregularity, spikiness, num_vertices)
    
    bg_color = 255 if label==0 else 0   # background opposite of polygon
    fg_color = 255 if label==1 else 0   # polygon fill
    
    img = Image.new('L', (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)
    draw.polygon(vertices, fill=fg_color)  # filled polygon
    
    return np.array(img, dtype=np.float32)/255.0, label

# -----------------------------
# Generate dataset
# -----------------------------
def generate_polygon_dataset(n_samples=1000, img_size=40, min_vertices=3, max_vertices=12):
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    for i in range(n_samples):
        num_vertices = random.randint(min_vertices, max_vertices)
        img, label = generate_polygon_image(img_size=img_size, num_vertices=num_vertices)
        X[i,0] = img
        y[i,0] = label
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Example usage & visualization
# -----------------------------
X_train, y_train = generate_polygon_dataset(n_samples=1000, img_size=100, min_vertices=10, max_vertices=25)

plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i,0], cmap='gray')
    plt.title(f"Label={int(y_train[i,0])}")
    plt.axis('off')
plt.show()

# ---------------------------
# 2. Flexible CNN
# ---------------------------
class CNN(nn.Module):
    def __init__(self, input_channels=1, conv_channels=[8, 16], kernel_size=11, output_dim=1):
        super().__init__()
        layers = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        # GAP reduces HxW to 1x1 per channel, output size = number of channels
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Flatten(),        # flatten Cx1x1 to C
            nn.Linear(in_ch, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = self.fc(x)
        return x
    
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Generate dataset (already in CPU)
# -----------------------------
X_train, y_train = generate_polygon_dataset(n_samples=1000, img_size=100, min_vertices=10, max_vertices=50)

# Move dataset to GPU
X_train = X_train.to(device)
y_train = y_train.to(device)

# ---------------------------
# Define CNN and move to GPU
# ---------------------------
conv_channels = [10, 10, 10]
model = CNN(conv_channels=conv_channels).to(device)

# ---------------------------
# Training
# ---------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    if loss == 0:
        break

# ---------------------------
# Visualize a sample
# ---------------------------
sample_img = X_train[0].unsqueeze(0)  # shape (1,1,H,W)
model.eval()
with torch.no_grad():
    _ = model(sample_img)

# ---------------------------
# Visualize feature maps on GPU
# ---------------------------
def visualize_feature_maps(model, X_sample, conv_layers=[0,2]):
    x = X_sample
    for i, layer in enumerate(model.conv_layers):
        x = layer(x)
        if i in conv_layers:
            n_channels = x.shape[1]
            plt.figure(figsize=(15,3))
            for c in range(n_channels):
                plt.subplot(1, n_channels, c+1)
                fmap = x[0,c].detach().cpu().numpy()  # move to CPU for plotting
                fmap = (fmap > 0.5).astype(np.float32)
                plt.imshow(fmap, cmap='gray')
                plt.axis('off')
                plt.title(f"Ch {c}")
            plt.suptitle(f"Feature maps after layer {i+1}")
            plt.show()

visualize_feature_maps(model, sample_img, conv_layers=[2*i for i in range(len(conv_channels))])

# ---------------------------
# Visualize conv filters
# ---------------------------
def visualize_conv_filters(model, threshold=None):
    """
    Visualize the filters (weights) of Conv2d layers.
    
    Parameters:
    - model: the CNN model
    - threshold: float or None. If set, only values with abs(weight) >= threshold are shown; others set to 0
    """
    for i, layer in enumerate(model.conv_layers):
        if isinstance(layer, nn.Conv2d):
            n_filters = layer.weight.shape[0]
            plt.figure(figsize=(15,3))
            for f in range(n_filters):
                filt = layer.weight[f,0].detach().cpu().numpy()
                if threshold is not None:
                    filt = np.where(np.abs(filt) >= threshold, filt, 0.0)
                plt.subplot(1, n_filters, f+1)
                plt.imshow(filt, cmap='gray', vmin=filt.min(), vmax=filt.max())
                plt.axis('off')
                plt.title(f"F {f}")
            plt.suptitle(f"Conv layer {i+1} filters")
            plt.show()

# Example usage with thresholding
visualize_conv_filters(model, threshold=0.5)

# ---------------------------
# Parameter count
# ---------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"{conv_channels} CNN has total parameters:", total_params)
print("Final loss:", loss.item())
