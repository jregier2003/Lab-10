import numpy as np
import matplotlib.pyplot as plt


def create_tridiagonal_matrix(size, below_diag, diag, above_diag):
    matrix = np.zeros((size, size))
    np.fill_diagonal(matrix, diag)
    np.fill_diagonal(matrix[1:], below_diag)
    np.fill_diagonal(matrix[:, 1:], above_diag)
    return matrix

size = 5
A = create_tridiagonal_matrix(size, below_diag=3, diag=1, above_diag=5)

matrix_test = create_tridiagonal_matrix(5, -2, 4, 1.5)
print("Generated Matrix:\n", matrix_test)

def initialize_wavepacket(sigma, k, grid):
    return np.exp(-grid**2 / (2 * sigma**2)) * np.cos(k * grid)

L = 5
n_space = 300
x_grid = np.linspace(-L / 2, L / 2, n_space)
sigma_val = 0.2
k_val = 35
wavepacket = initialize_wavepacket(sigma_val, k_val, x_grid)

plt.plot(x_grid, wavepacket)
plt.xlabel("x")
plt.ylabel("a(x, 0)")
plt.title("Initial Wavepacket")
plt.show()


def compute_spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

test_matrix = create_tridiagonal_matrix(5, -2, 4, 1.5)
radius = compute_spectral_radius(test_matrix)
print("Spectral Radius:", radius)


