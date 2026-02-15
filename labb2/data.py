import numpy as np
import random
 
# Generate class A (two clusters)
classA = np.concatenate(
    (
        np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
        np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
    )
)

# Generate class B (single cluster)
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

# Combine inputs
inputs = np.concatenate((classA, classB))

# Create targets (+1 for class A, -1 for class B)
targets = np.concatenate(
    (
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    )
)

# Number of samples
N = inputs.shape[0]

# Shuffle dataset
permute = list(range(N))
random.shuffle(permute)

inputs = inputs[permute, :]
targets = targets[permute]

