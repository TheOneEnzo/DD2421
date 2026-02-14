import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from data import *

#linear kernel x.T * y
def linear_kernel(x,y):
    #verifiera shape?
    return x.reshape(1,-1) @ y

def objective(alpha): #effectively implementing the expression that should be minimized, in our case equation (4)
    
    return scalar

start = np.zeros(N)

alpha = np.array([0,5])
gamma = np.array([2,1])
print(alpha.shape)
#ret = minimize(objective(alpha), start, bounds=[(0, None) for b in range(N)], constraints=XC)
#alpha = ret['x']

print(f"Linear Kernel {linear_kernel(alpha, gamma)}")

"""
# Plot class A (blue points)
plt.plot(
    [p[0] for p in classA],
    [p[1] for p in classA],
    'b.'
)

# Plot class B (red points)
plt.plot(
    [p[0] for p in classB],
    [p[1] for p in classB],
    'r.'
)

plt.axis('equal')          # Force same scale on both axes
plt.savefig('svmplot.pdf') # Save plot to file
plt.show()                 # Display plot


# Create grid for contour plot
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([
    [indicator(x, y) for x in xgrid]
    for y in ygrid
])

# Draw contour lines
plt.contour(
    xgrid,
    ygrid,
    grid,
    levels=(-1.0, 0.0, 1.0),
    colors=('red', 'black', 'blue'),
    linewidths=(1, 3, 1)
)

plt.show()
"""