import numpy as np

# Problem setup
T = 1.0          # Terminal time
N = 100          # Number of time steps
dt = T / N       # Time step size
d = 1            # Dimension of Brownian motion
M = 10000        # Number of Monte Carlo samples

# Define functions
def sigma(t, x):
    return np.ones((M, d))  # Example: constant volatility

def sigma_inv(t, x):
    return np.ones((M, d))  # Inverse of sigma (identity for constant sigma)

def H(t, x, z):
    return -0.5 * np.sum(z**2, axis=1)  # Example Hamiltonian

def g(x):
    return np.sum(x**2, axis=1)  # Terminal condition

# Initialize arrays
X = np.zeros((N + 1, M, d))
Y = np.zeros((N + 1, M))
Z = np.zeros((N, M, d))
dW = np.random.normal(0, np.sqrt(dt), size=(N, M, d))

# Simulate forward SDE (Euler-Maruyama)
for n in range(N):
    X[n + 1] = X[n] + sigma(n * dt, X[n]) * dW[n]

# Terminal condition
Y[N] = g(X[N])

# Backward recursion for BSDE
for n in reversed(range(N)):
    # Estimate conditional expectations using Monte Carlo
    EY = np.mean(Y[n + 1], axis=0)
    EYdW = np.mean(Y[n + 1][:, None] * dW[n], axis=0)

    # Compute Z and Y
    Z[n] = (1 / dt) * sigma_inv(n * dt, X[n]) * EYdW
    Y[n] = EY + H(n * dt, X[n], Z[n]) * dt

# Output approximation of Y_0
print("Approximate Y_0:", np.mean(Y[0]))
