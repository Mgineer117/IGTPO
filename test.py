import numpy as np

probability_history = np.array([0.000000000000004, 0.0, 0.00000000000005])
# probability_history = np.array([0.23, 0.75, 0.5])
# probability_history = np.array([0.00023, 0.00075, 0.0005])

# Step 1: Normalize to [0, 1] range
weights = (probability_history - probability_history.min()) / (
    probability_history.max() - probability_history.min() + 1e-8
)
weights = weights / (weights.sum() + 1e-8)
print(weights)
