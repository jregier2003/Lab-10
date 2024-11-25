import numpy as np
import matplotlib.pyplot as plt
#Link to github Repository: https://github.com/jregier2003/RegierJacob_Lab10_commits.git


def create_tridiagonal_matrix(size, below_diag, diag, above_diag):
    """
    Creates a tridiagonal matrix with specified diagonal and off-diagonal values.

    Parameters:
    size (int): Dimension of the matrix (NxN).
    below_diag (float): Value below the main diagonal.
    diag (float): Value on the main diagonal.
    above_diag (float): Value above the main diagonal.

    Returns:
    numpy.ndarray: The generated tridiagonal matrix.
    """
    matrix = np.zeros((size, size))
    np.fill_diagonal(matrix, diag)
    np.fill_diagonal(matrix[1:], below_diag)
    np.fill_diagonal(matrix[:, 1:], above_diag)
    return matrix


#Test case for the tridiagonal matrix generation
matrix_test = create_tridiagonal_matrix(5, -2, 4, 1.5)
print("Generated Matrix:\n", matrix_test)


def initialize_wavepacket(sigma, k, grid):
    """
    Generates the initial wavepacket based on a Gaussian and cosine function.

    Parameters:
    sigma (float): Standard deviation of the Gaussian envelope.
    k (float): Wavenumber of the cosine function.
    grid (numpy.ndarray): Spatial grid for the wavepacket.

    Returns:
    numpy.ndarray: Initial wavepacket values at grid points.
    """
    return np.exp(-grid**2 / (2 * sigma**2)) * np.cos(k * grid)


# Define parameters of the wavepacket
L = 5
n_space = 300
x_grid = np.linspace(-L / 2, L / 2, n_space)
sigma_val = 0.2
k_val = 35
wavepacket = initialize_wavepacket(sigma_val, k_val, x_grid)

# Plot the wavepacket
plt.plot(x_grid, wavepacket)
plt.xlabel("x")
plt.ylabel("a(x, 0)")
plt.title("Initial Wavepacket")
plt.show()


def compute_spectral_radius(matrix):
    """
    Computes the spectral radius (maximum absolute eigenvalue) of a matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix to compute eigenvalues.

    Returns:
    float: The spectral radius of the matrix.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))


# Compute spectral radius for a small test matrix
test_matrix = create_tridiagonal_matrix(5, -2, 4, 1.5)
radius = compute_spectral_radius(test_matrix)
print("Spectral Radius:", radius)

# Calculate spectral radius for a larger matrix
n = 10
matrix = create_tridiagonal_matrix(n, -1, 2, -1)
radius = compute_spectral_radius(matrix)
print(f"Spectral Radius of a {n}x{n} matrix:", radius)

