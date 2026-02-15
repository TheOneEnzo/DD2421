import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from data import inputs, targets, classA, classB, N

# -----------------------------
# 1) Kernel
# -----------------------------
def linear_kernel(x, y):
    # scalar dot product
    return float(np.dot(x, y))

def polynomial_kernel(x,y,p):
    return float((np.dot(x,y) + 1)**p)

def RBF_kernel(x, y, sigma):
    d2 = np.linalg.norm(x - y)**2
    return float(np.exp(-d2 / (2 * sigma**2)))


# -----------------------------
# 2) Precompute P matrix
#    P_ij = t_i t_j K(x_i, x_j)
# -----------------------------
Kmat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Kmat[i, j] = linear_kernel(inputs[i], inputs[j])

P = (targets[:, None] * targets[None, :]) * Kmat

# -----------------------------
# 3) Dual objective (eq 4)
#    1/2 a^T P a - sum(a)
# -----------------------------
def objective(alpha):
    return 0.5 * alpha @ P @ alpha - np.sum(alpha)

# -----------------------------
# 4) Equality constraint (eq 10)
#    sum_i alpha_i t_i = 0
# -----------------------------
def zerofun(alpha):
    return np.dot(alpha, targets)

XC = {'type': 'eq', 'fun': zerofun}

# -----------------------------
# 5) Solve for alpha with bounds
#    0 <= alpha_i <= C
# -----------------------------
C = 0.5
B = [(0.0, C) for _ in range(N)]
start = np.zeros(N)

ret = minimize(objective, start, bounds=B, constraints=XC)

if not ret.success:
    raise RuntimeError(f"Optimizer failed: {ret.message}")

alpha = ret.x

# -----------------------------
# 6) Extract support vectors
# -----------------------------
eps = 1e-5
sv_idx = np.where(alpha > eps)[0]

sv_alpha = alpha[sv_idx]
sv_x = inputs[sv_idx]
sv_t = targets[sv_idx]

print(f"Number of support vectors: {len(sv_idx)}")

# -----------------------------
# 7) Compute b (eq 7)
# Use a point on the margin: 0 < alpha_i < C (if possible)
# -----------------------------
margin_idx = np.where((alpha > eps) & (alpha < C - eps))[0]
if len(margin_idx) > 0:
    s_idx = margin_idx[0]
else:
    # fallback: use any SV (may be slightly less stable with slack)
    s_idx = sv_idx[0]

def decision_no_bias(x):
    # sum_i alpha_i t_i K(x, x_i) over support vectors
    return np.sum(sv_alpha * sv_t * np.array([linear_kernel(x, xi) for xi in sv_x]))

b = decision_no_bias(inputs[s_idx]) - targets[s_idx]
print(f"b = {b:.6f}")

# -----------------------------
# 8) Indicator function (eq 6)
# -----------------------------
def indicator(x):
    return decision_no_bias(x) - b

# -----------------------------
# 9) Plot data + decision boundary + margins
# -----------------------------
# Plot original classes (from data generation)
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label='Class +1')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label='Class -1')

# Highlight support vectors
plt.plot(sv_x[:, 0], sv_x[:, 1], 'ko', fillstyle='none', markersize=10, label='Support Vectors')

plt.axis('equal')

# Grid for contour
xgrid = np.linspace(-5, 5, 200)
ygrid = np.linspace(-4, 4, 200)
grid = np.array([[indicator(np.array([x, y])) for x in xgrid] for y in ygrid])

# Contours at -1, 0, +1
plt.contour(
    xgrid, ygrid, grid,
    levels=(-1.0, 0.0, 1.0),
    colors=('red', 'black', 'blue'),
    linewidths=(1, 3, 1)
)

plt.legend()
plt.title("SVM (dual) with linear kernel: decision boundary and margins")
plt.show()
