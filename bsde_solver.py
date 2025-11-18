import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


class BSDESolver:
    """
    Solver for Backward Stochastic Differential Equations (BSDEs)
    using the discrete-time scheme with conditional expectation approximation.
    """

    def __init__(self, T, N, d, n_samples=10000, n_test=5000,
                 cost_func=None, terminal_func=None, diffusion_func=None):
        """
        Initialize BSDE solver.

        Parameters:
        -----------
        T : float
            Terminal time
        N : int
            Number of time steps
        d : int
            Dimension of the state variable X
        n_samples : int
            Number of Monte Carlo samples for training
        n_test : int
            Number of test samples
        cost_func : callable, optional
            Cost function F(t, X, Z). If None, uses default F = -0.5|Z|^2
            Signature: F(t, X, Z) -> array of shape (n_samples, 1)
        terminal_func : callable, optional
            Terminal condition g(X_T). If None, uses default g(X) = |X|^2
            Signature: g(X) -> array of shape (n_samples, 1)
        diffusion_func : callable, optional
            Diffusion coefficient sigma(t, X). If None, uses constant diffusion
            Signature: sigma(t, X) -> array of shape (n_samples, d)
        """
        self.T = T
        self.N = N
        self.d = d
        self.dt = T / N
        self.n_samples = n_samples
        self.n_test = n_test
        self.time_grid = np.linspace(0, T, N + 1)

        # Custom functions
        self._cost_func = cost_func
        self._terminal_func = terminal_func
        self._diffusion_func = diffusion_func

        # Storage for solution
        self.X_paths = None
        self.Y_values = None
        self.Z_values = None

    def sigma(self, t, X):
        """
        Diffusion coefficient for the forward process.
        Default: constant diffusion
        """
        if self._diffusion_func is not None:
            return self._diffusion_func(t, X)
        return np.ones_like(X)

    def F(self, t, X, Z):
        """
        Generator function F(t, X, Z).
        Default: F = -0.5 * |Z|^2
        """
        if self._cost_func is not None:
            return self._cost_func(t, X, Z)
        return -0.5 * np.sum(Z ** 2, axis=-1, keepdims=True)

    def g(self, X):
        """
        Terminal condition g(X_T).
        Default: g(X) = sum(X^2)
        """
        if self._terminal_func is not None:
            return self._terminal_func(X)
        return np.sum(X ** 2, axis=-1, keepdims=True)

    def generate_brownian_increments(self, n_paths):
        """Generate Brownian motion increments."""
        return np.random.randn(n_paths, self.N, self.d) * np.sqrt(self.dt)

    def simulate_forward_process(self, n_paths, dW=None):
        """
        Simulate the forward SDE: dX_t = sigma(t, X_t) dW_t

        Returns:
        --------
        X : array of shape (n_paths, N+1, d)
            Simulated paths
        dW : array of shape (n_paths, N, d)
            Brownian increments
        """
        if dW is None:
            dW = self.generate_brownian_increments(n_paths)

        X = np.zeros((n_paths, self.N + 1, self.d))
        X[:, 0, :] = np.random.uniform(-1,1,[n_paths,self.d])  # Start at origin

        for n in range(self.N):
            t = self.time_grid[n]
            sigma_val = self.sigma(t, X[:, n, :])
            X[:, n + 1, :] = X[:, n, :] + sigma_val * dW[:, n, :]

        return X, dW

    def compute_conditional_expectation(self, X_current, Y_next, basis_type='polynomial', degree=3):
        """
        Compute E[Y_{n+1} | X_n] using regression on basis functions.

        Parameters:
        -----------
        X_current : array of shape (n_paths, d)
            Current state values
        Y_next : array of shape (n_paths, 1)
            Next time step Y values
        basis_type : str
            Type of basis functions ('polynomial' or 'gaussian')
        degree : int
            Degree of polynomial basis
        """
        n_paths = X_current.shape[0]

        if basis_type == 'polynomial':
            # Create polynomial basis
            basis = [np.ones((n_paths, 1))]

            # Linear terms
            for i in range(self.d):
                basis.append(X_current[:, i:i + 1])

            # Quadratic terms
            if degree >= 2:
                for i in range(self.d):
                    basis.append(X_current[:, i:i + 1] ** 2)
                for i in range(self.d):
                    for j in range(i + 1, self.d):
                        basis.append(X_current[:, i:i + 1] * X_current[:, j:j + 1])

            # Cubic terms
            if degree >= 3:
                for i in range(self.d):
                    basis.append(X_current[:, i:i + 1] ** 3)

            Phi = np.hstack(basis)

        # Least squares regression
        theta = np.linalg.lstsq(Phi, Y_next, rcond=None)[0]
        Y_pred = Phi @ theta

        return Y_pred, (Phi, theta)

    def compute_Z(self, X_current, Y_next, dW_next, basis_params):
        """
        Compute Z_n = (1/dt) * E[Y_{n+1} * dW_{n+1} | X_n]
        """
        Phi, theta = basis_params

        # Compute Y_{n+1} * dW_{n+1}
        Y_dW = Y_next * dW_next  # shape: (n_paths, d)

        # Regress each component
        Z = np.zeros((X_current.shape[0], self.d))
        for i in range(self.d):
            theta_i = np.linalg.lstsq(Phi, Y_dW[:, i:i + 1], rcond=None)[0]
            Z[:, i:i + 1] = (Phi @ theta_i) / self.dt

        return Z

    def solve(self, verbose=True):
        """
        Solve the BSDE using backward recursion.
        """
        # Generate forward paths
        X, dW = self.simulate_forward_process(self.n_samples)
        self.X_paths = X

        # Initialize storage
        Y = np.zeros((self.n_samples, self.N + 1, 1))
        Z = np.zeros((self.n_samples, self.N, self.d))

        # Terminal condition
        Y[:, -1, :] = self.g(X[:, -1, :])

        # Backward iteration
        for n in range(self.N - 1, -1, -1):
            t = self.time_grid[n]

            if verbose and n % max(1, self.N // 10) == 0:
                print(f"Time step {n}/{self.N}, t={t:.3f}")

            # Compute E[Y_{n+1} | X_n]
            E_Y_next, basis_params = self.compute_conditional_expectation(
                X[:, n, :], Y[:, n + 1, :]
            )

            # Compute Z_n
            Z[:, n, :] = self.compute_Z(
                X[:, n, :], Y[:, n + 1, :], dW[:, n, :], basis_params
            )

            # Compute Y_n using the BSDE scheme
            F_val = self.F(t, X[:, n, :], Z[:, n, :])
            Y[:, n, :] = E_Y_next + F_val * self.dt

        self.Y_values = Y
        self.Z_values = Z

        # Compute Y_0 (initial value)
        Y_0 = np.mean(Y[:, 0, 0])

        return Y_0

    def plot_results(self):
        """Generate visualization of the BSDE solution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Sample paths of X
        ax = axes[0, 0]
        n_plot = min(100, self.n_samples)
        for i in range(n_plot):
            ax.plot(self.time_grid, self.X_paths[i, :, 0],
                    alpha=0.3, linewidth=0.5, color='blue')
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('$X_t$ (first component)', fontsize=11)
        ax.set_title('Sample Paths of Forward Process $X_t$', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 2: Y values over time
        ax = axes[0, 1]
        Y_mean = np.mean(self.Y_values[:, :, 0], axis=0)
        Y_std = np.std(self.Y_values[:, :, 0], axis=0)
        ax.plot(self.time_grid, Y_mean, 'r-', linewidth=2, label='Mean')
        ax.fill_between(self.time_grid,
                        Y_mean - 2 * Y_std,
                        Y_mean + 2 * Y_std,
                        alpha=0.3, color='red', label='±2 std')
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('$Y_t$', fontsize=11)
        ax.set_title('Solution $Y_t$ Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 3: Distribution of Y at different times
        ax = axes[1, 0]
        times_to_plot = [0, self.N // 2, self.N]
        colors = ['blue', 'green', 'red']
        for idx, (t_idx, color) in enumerate(zip(times_to_plot, colors)):
            ax.hist(self.Y_values[:, t_idx, 0], bins=50, alpha=0.5,
                    color=color, label=f't={self.time_grid[t_idx]:.2f}', density=True)
        ax.set_xlabel('$Y_t$', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Distribution of $Y_t$ at Different Times', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 4: Z values over time (first component)
        ax = axes[1, 1]
        Z_mean = np.mean(self.Z_values[:, :, 0], axis=0)
        Z_std = np.std(self.Z_values[:, :, 0], axis=0)
        ax.plot(self.time_grid[:-1], Z_mean, 'g-', linewidth=2, label='Mean')
        ax.fill_between(self.time_grid[:-1],
                        Z_mean - 2 * Z_std,
                        Z_mean + 2 * Z_std,
                        alpha=0.3, color='green', label='±2 std')
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('$Z_t$ (first component)', fontsize=11)
        ax.set_title('Control Process $Z_t$ Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bsde_solution.png', dpi=150, bbox_inches='tight')
        plt.show()

        return fig


# Example usage with a specific BSDE
class ExampleBSDE(BSDESolver):
    """
    Example: Solve BSDE with F(t,X,Z) = -0.5|Z|^2 and g(X) = |X|^2
    This corresponds to a nonlinear pricing problem.
    """

    def sigma(self, t, X):
        """Constant volatility."""
        return 0.3 * np.ones_like(X)

    def F(self, t, X, Z):
        """Quadratic generator."""
        return -0.5 * np.sum(Z ** 2, axis=-1, keepdims=True)

    def g(self, X):
        """Quadratic terminal payoff."""
        return np.sum(X ** 2, axis=-1, keepdims=True)


if __name__ == "__main__":
    print("=" * 60)
    print("BSDE Solver - Discrete Time Approximation")
    print("=" * 60)
    print()

    # Setup parameters
    T = 1.0  # Terminal time
    N = 50  # Number of time steps
    d = 1  # Dimension of state space
    n_samples = 5000  # Number of Monte Carlo paths

    print(f"Parameters:")
    print(f"  Terminal time T = {T}")
    print(f"  Time steps N = {N}")
    print(f"  State dimension d = {d}")
    print(f"  MC samples = {n_samples}")
    print()

    # Example 1: Using the ExampleBSDE class
    print("=" * 60)
    print("EXAMPLE 1: Using ExampleBSDE class")
    print("=" * 60)
    print("Initializing BSDE solver...")
    bsde1 = ExampleBSDE(T=T, N=N, d=d, n_samples=n_samples)

    print("\nSolving BSDE backward in time...")
    print("-" * 60)
    Y_0_1 = bsde1.solve(verbose=True)
    print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"SOLUTION AT t=0:")
    print(f"  Y_0 = {Y_0_1:.6f}")
    print(f"{'=' * 60}")

    # Generate plots
    print("\nGenerating visualizations...")
    bsde1.plot_results()
    print("Plots saved to 'bsde_solution.png'")

    # Example 2: Using custom functions directly
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Using custom cost and terminal functions")
    print("=" * 60)


    # Define custom functions
    def my_cost_func(t, X, Z):
        """Custom cost: F(t,X,Z) = -|Z|^2 + sin(t)*X"""
        return -np.sum(Z ** 2, axis=-1, keepdims=True) + np.sin(t) * np.sum(X, axis=-1, keepdims=True)


    def my_terminal_func(X):
        """Custom terminal condition: g(X) = exp(-|X|^2/2)"""
        return np.exp(-0.5 * np.sum(X ** 2, axis=-1, keepdims=True))


    def my_diffusion_func(t, X):
        """Custom diffusion: sigma(t,X) = 0.2 * (1 + 0.5*sin(t))"""
        return 0.2 * (1 + 0.5 * np.sin(t)) * np.ones_like(X)


    print("\nCustom Functions:")
    print("  F(t,X,Z) = -|Z|^2 + sin(t)*X")
    print("  g(X) = exp(-|X|^2/2)")
    print("  sigma(t,X) = 0.2*(1 + 0.5*sin(t))")
    print()

    bsde2 = BSDESolver(
        T=T, N=N, d=d, n_samples=n_samples,
        cost_func=my_cost_func,
        terminal_func=my_terminal_func,
        diffusion_func=my_diffusion_func
    )

    print("Solving custom BSDE backward in time...")
    print("-" * 60)
    Y_0_2 = bsde2.solve(verbose=False)
    print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"SOLUTION AT t=0:")
    print(f"  Y_0 = {Y_0_2:.6f}")
    print(f"{'=' * 60}")

    print("\nDone!")