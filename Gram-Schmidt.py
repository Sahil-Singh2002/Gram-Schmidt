import numpy as np
import sympy

def gram_schmidt_np(V):
    # Validate the input
    if not isinstance(V, list) or any(not isinstance(v, np.ndarray) or v.ndim != 1 or v.size != len(V) for v in V):
        raise ValueError("Invalid input. Expected a list of 1-D NumPy arrays with consistent lengths.")

    W = [V[0]]  # Initial condition

    for i in range(1, len(V)):
        summation = np.zeros_like(V[i])
        for j in range(i):
            dot_product = np.dot(V[i], W[j])
            norm_squared = np.dot(W[j], W[j])
            summation += (dot_product / norm_squared) * W[j]
        W.append(V[i] - summation)

    for i in range(len(W)):
        if not np.allclose(W[i], 0):  # Check if the vector is not a zero vector
            W[i] = W[i] / np.linalg.norm(W[i])
        else:
            W[i] = np.zeros_like(W[i])

    return W


def gram_schmidt_sp(V):
    # Validate the input
    if not isinstance(V, sympy.Matrix) or V.rows != V.cols:
        raise ValueError("Invalid input. Expected a square SymPy matrix.")

    v = [V.col(i).T for i in range(V.cols)]
    W = [v[0]]  # Initial condition

    for i in range(1, len(v)):
        summation = sympy.zeros(1, V.cols)
        for j in range(i):
            dot_product = v[i].dot(W[j])
            norm_squared = W[j].dot(W[j])
            summation += (dot_product / norm_squared) * W[j]
        W.append(v[i] - summation)

    for i in range(len(W)):
        if not W[i].is_zero:  # Check if the vector is not a zero vector
            W[i] = W[i] / W[i].norm()
        else:
            W[i] = sympy.zeros(1, V.cols)

    return sympy.Matrix(W)
