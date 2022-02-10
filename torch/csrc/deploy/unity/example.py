import numpy as np
import scipy
from scipy import linalg

print("Hello, torch::deploy unity!")
print(f"np.random.rand(5): {np.random.rand(5)}")
print(f"scipy {scipy}")
mat_a = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 1, 0], [1, 3, 3, 1]])
mat_b = linalg.inv(mat_a)
print(mat_b)
