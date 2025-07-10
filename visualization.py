import matplotlib.pyplot as plt
import numpy as np
import torch

# Hyperparameters
alpha = 0.05  # pseudo-loss learning rate
beta = 1.0  # meta learning rate
num_steps = 10
meta_steps = 500
tol = 1e-3

pseudo_optima = (0, -2)


# Loss functions
def pseudo_loss_fn(x):
    return (
        torch.norm(x[0] - pseudo_optima[0]) ** 2
        + torch.norm(x[1] - pseudo_optima[1]) ** 2
    )


def loss_fn(x):
    return torch.norm(x) ** 2


# Compute meta gradient and record all pseudo-loss paths
def compute_meta_gradient(theta_init):
    theta_list = []
    grad_list = []

    theta = theta_init.clone().detach().requires_grad_(True)
    theta_list.append(theta)

    for _ in range(num_steps):
        loss = pseudo_loss_fn(theta)
        grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
        grad_list.append(grad)
        theta = theta - alpha * grad
        theta_list.append(theta)

    # Final meta loss and gradient
    loss_meta = loss_fn(theta)
    v_final = torch.autograd.grad(loss_meta, theta, retain_graph=True)[0]

    # Backward pass (reverse-mode Hessian-vector product)
    v = v_final
    for i in reversed(range(num_steps)):
        H_v = torch.autograd.grad(
            grad_list[i], theta_list[i], grad_outputs=v, retain_graph=True
        )[0]
        v = v - alpha * H_v

    return v.detach(), theta_list, v_final.detach()


# Initialize meta optimization
theta0 = torch.tensor([6.0, -6.0])
meta_trajectory = [theta0.numpy()]
meta_grads = []
local_grads = []
all_pseudo_trajectories = []

# Meta optimization loop
for i in range(meta_steps):
    meta_grad, theta_list, grad_theta_n = compute_meta_gradient(theta0)
    if torch.linalg.norm(meta_grad, 2) < tol:
        meta_steps = i
        break
    meta_grads.append(meta_grad.numpy())
    local_grads.append(grad_theta_n.numpy())
    all_pseudo_trajectories.append(
        torch.stack([t.detach() for t in theta_list]).numpy()
    )
    theta0 = theta0 - beta * meta_grad
    meta_trajectory.append(theta0.numpy())
print(f"Meta steps: {meta_steps}")
meta_trajectory = np.stack(meta_trajectory)
meta_grads = np.stack(meta_grads)

# Prepare background contour
x = np.linspace(-8, 8, 100)
y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Plotting
plt.figure(figsize=(12, 12))
plt.contour(X, Y, -Z, levels=25, cmap="viridis")

# Plot all pseudo-loss trajectories
for i, traj in enumerate(all_pseudo_trajectories):
    start = traj[0]
    end = traj[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    plt.quiver(
        start[0],
        start[1],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
        alpha=0.8 - 0.4 * (i / meta_steps),
        linewidth=2,
        label="Inner-level progression" if i == 0 else None,
    )


# Plot meta-gradient descent path
X = meta_trajectory[:-1, 0]  # starting x-coordinates
Y = meta_trajectory[:-1, 1]  # starting y-coordinates
U = meta_trajectory[1:, 0] - meta_trajectory[:-1, 0]  # dx
V = meta_trajectory[1:, 1] - meta_trajectory[:-1, 1]  # dy

plt.quiver(
    X,
    Y,
    U,
    V,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="b",
    alpha=0.8,
    linewidth=2,
    label="Outer-level progression",
)


# Points
# Optimal pseudo-loss point (distinct yellow with black border)
plt.scatter(
    pseudo_optima[0],
    pseudo_optima[1],
    s=850,
    color="red",
    edgecolors="black",
    linewidths=1.5,
    marker="*",
    label="Intrinsic optima",
    zorder=5,
)

# Optimal final loss (green star with black edge)
plt.scatter(
    0,
    0,
    s=850,
    color="green",
    edgecolors="black",
    linewidths=1.5,
    marker="*",
    label="Extrinsic optima",
    zorder=5,
)

# # Quiver: meta-gradient direction
# for i in range(meta_steps):
#     grad = -meta_grads[i]
#     norm = 1e-2 * np.linalg.norm(grad)
#     plt.quiver(
#         meta_trajectory[i, 0],
#         meta_trajectory[i, 1],
#         grad[0],
#         grad[1],
#         angles="xy",
#         scale_units="xy",
#         # scale=1.0 / (norm + 1e-6),  # prevent divide-by-zero
#         color="purple",
#         # width=0.25 * norm,
#         label=(
#             r"$\nabla_{\tilde{\theta}^{(0)}} \mathcal{J}(\tilde{\theta}^{(N)})$"
#             if i == 0
#             else None
#         ),
#     )


# # Final ∇θₙ L_meta
# for i in range(meta_steps):
#     grad_local = -local_grads[i]
#     norm_local = 1e-2 * np.linalg.norm(grad_local)
#     plt.quiver(
#         all_pseudo_trajectories[i][-1, 0],
#         all_pseudo_trajectories[i][-1, 1],
#         grad_local[0],
#         grad_local[1],
#         angles="xy",
#         scale_units="xy",
#         color="green",
#         # width=0.08 * norm_local,
#         label=(
#             r"$\nabla_{\tilde{\theta}^{(N)}} \mathcal{J}(\tilde{\theta}^{(N)})$"
#             if i == 0
#             else None
#         ),
#     )


# Starting θ₀ (large blue star with white border)
plt.scatter(
    meta_trajectory[0, 0],
    meta_trajectory[0, 1],
    s=500,
    color="lightpink",
    edgecolors="grey",
    linewidths=3,
    label=r"Start $\theta$",
    zorder=6,
)

# Final θ₀ after meta-updates (blue square with black edge)
plt.scatter(
    meta_trajectory[-1, 0],
    meta_trajectory[-1, 1],
    s=500,
    color="skyblue",
    edgecolors="grey",
    linewidths=3,
    label=r"Final $\theta$",
    zorder=5,
)

# Labels and legends
plt.title(f"IRPO with sub-iterations {num_steps}", fontsize=42)
plt.xlabel(r"$\theta[0]$", fontsize=28)
plt.ylabel(r"$\theta[1]$", fontsize=28)
# Tick label sizes
plt.tick_params(axis="x", labelsize=26)
plt.tick_params(axis="y", labelsize=26)
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.6)

# Place legend above the figure
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, fontsize=24)

# Show the figure
# plt.tight_layout()
plt.show()
