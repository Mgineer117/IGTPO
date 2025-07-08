import numpy as np

probability_history = np.array([0.0001, 0.01, 0.007])
# probability_history = np.array([0.23, 0.75, 0.5])
# probability_history = np.array([0.00023, 0.00075, 0.0005])

# weights = probability_history
weights = np.random.rand(len(probability_history))
# print(weights)
weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
weights = (weights) / ((weights).sum() + 1e-8)

print(weights)
