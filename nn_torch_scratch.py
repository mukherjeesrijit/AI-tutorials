import torch

# Step 1: Model Creation
# Reminder: No nn.Module is here.

class MyModel:
    def __init__(self):
        # Leaf tensors: requires_grad=True, .grad=None
        self.W1 = torch.randn(2, 3, requires_grad=True) # input 2 → hidden 3
        self.b1 = torch.zeros(3, requires_grad=True)
        self.W2 = torch.randn(3, 1, requires_grad=True) # hidden 3 → output 1
        self.b2 = torch.zeros(1, requires_grad=True)
        # Store parameters in a list for easy optimizer step
        self.params = [self.W1, self.b1, self.W2, self.b2]

    # Step 2: Forward Pass
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1     # Linear1
        self.a1 = torch.relu(self.z1)      # Activation
        self.z2 = self.a1 @ self.W2 + self.b2  # Linear2
        return self.z2

# Step 3: Input
xb = torch.tensor([[1.0, 2.0]]) # shape (1,2)
yb = torch.tensor([[1.0]])    # target

# Instantiate model
model = MyModel()

# Step 4: Loss Computation (custom MSE loss)
def mse_loss(pred, target):
    # Returns a scalar tensor connected to graph
    return ((pred - target)**2).mean()

# Step 5: Custom optimizer (SGD)
# No optimizer is called here.

class SGD:
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    # Update step using .grad
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    # Step 6: Zero gradients
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    optimizer = SGD(model.params, lr=0.1)

# Training loop for 5 steps
for i in range(5):
    # Step 7: Forward Pass
    output = model.forward(xb)

    # Step 8: Loss Computation
    loss = mse_loss(output, yb)
    print(f"Step {i}, Loss: {loss.item()}")

    # Step 9: Backward Pass [The Magic Happens here!!!! 🔴]
    loss.backward() # populates .grad for leaf tensors (W1, b1, W2, b2)

    # Step 10: Optimizer Step
    optimizer.step()

    # Step 11: Zero Gradients
    optimizer.zero_grad()
