import os
import time

import psutil
import torch

# Learning rate
alpha = 0.1


# Define loss functions
def loss_fn(x):
    return torch.linalg.norm(x - 2) ** 2


n = 64 * 64
# ===============================
# HVP-based approach (efficient)
# ===============================
start_time_hvp = time.time()

# θ_0
theta_0 = torch.randn(n, requires_grad=True)
# theta_0 = torch.tensor([0.0, 0.1, 2.0, 4.0, 0.5], requires_grad=True)

# Step 1: θ_1 = θ_0 - α ∇L_0
loss_0 = loss_fn(theta_0)
grad_0 = torch.autograd.grad(loss_0, theta_0, create_graph=True)[0]
theta_1 = theta_0 - alpha * grad_0

# Step 2: θ_2 = θ_1 - α ∇L_1
loss_1 = loss_fn(theta_1)
grad_1 = torch.autograd.grad(loss_1, theta_1, create_graph=True)[0]
theta_2 = theta_1 - alpha * grad_1

# Final meta-loss
loss_meta = loss_fn(theta_2)

# Compute ∇L_meta w.r.t. θ_2
v2 = torch.autograd.grad(loss_meta, theta_2)[0]

# Backprop through step 2: v1 = v2 · (I - α H1)
H1v = torch.autograd.grad(grad_1, theta_1, grad_outputs=v2)[0]
v1 = v2 - alpha * H1v

# Backprop through step 1: v0 = v1 · (I - α H0)
H0v = torch.autograd.grad(grad_0, theta_0, grad_outputs=v1)[0]
v0_hvp = v1 - alpha * H0v

hvp_time = time.time() - start_time_hvp

# =================================
# Full Hessian-based computation
# =================================
start_time_hessian = time.time()

# Recreate θ_0
theta_0 = theta_0.detach().clone().requires_grad_(True)
# theta_0 = torch.tensor([0.0, 0.1, 2.0, 4.0, 0.5], requires_grad=True)

# Step 1: θ_1 = θ_0 - α ∇L_0
loss_0 = loss_fn(theta_0)
grad_0 = torch.autograd.grad(loss_0, theta_0, create_graph=True)[0]
theta_1 = theta_0 - alpha * grad_0

# Step 2: θ_2 = θ_1 - α ∇L_1
loss_1 = loss_fn(theta_1)
grad_1 = torch.autograd.grad(loss_1, theta_1, create_graph=True)[0]
theta_2 = theta_1 - alpha * grad_1

# Final meta-loss
loss_meta = loss_fn(theta_2)

# Compute dL_meta/dθ_2
grad_meta = torch.autograd.grad(loss_meta, theta_2, create_graph=True)[0]

# Compute full Hessians
H1 = torch.autograd.functional.hessian(lambda x: loss_fn(x), theta_1)
H0 = torch.autograd.functional.hessian(lambda x: loss_fn(x), theta_0)


# Apply chain rule: v0 = (I - αH0)(I - αH1) ∇L_meta
I = torch.eye(H0.size(0))
v0_hessian = ((I - alpha * H0) @ (I - alpha * H1) @ grad_meta.unsqueeze(-1)).squeeze()


hessian_time = time.time() - start_time_hessian

# Memory usage in MB
mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


print(
    f"HVP-based v0: {v0_hvp}, Hessian-based v0: {v0_hessian}, "
    f"Difference: {torch.linalg.norm(v0_hvp - v0_hessian)}, "
    f"HVP time: {hvp_time:.4f}s, Hessian time: {hessian_time:.4f}s, "
    f"Memory usage: {mem_usage:.2f}MB"
)
