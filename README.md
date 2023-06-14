$$\textbf{Gram-Schmidt Process}$$

The Gram-Schmidt process is an algorithm that can be used to construct an orthonormal set of vectors { $w_i$ }, where each vector is pairwise orthogonal and has a length of 1.

Given a set of n linearly independent vectors { ${v_i | i = 1, ..., n}$ }in $\mathbb{R}^n$, the Gram-Schmidt process will produce a set of n orthonormal vectors { ${w_i | i = 1, ..., n}$ } in $\mathbb{R}^n$.

If the input vectors are not all linearly independent, the output will consist of an orthonormal basis for the linear span of the vectors { $v_i$ } along with several zero-vectors.

The method can be summarized by the following instructions:

1. Construct an orthogonal set of vectors { $w_i$ } using the formula: 
   - $$w_1 = v_1 ,$$
   - $$w_i = v_i - \sum_{j=1}^{i-1} \frac{{(v_i \cdot w_j)}}{{\lVert w_j \rVert^2}} \cdot w_j \quad \text{for } i > 1$$


2. If the vector wi obtained from the previous step is not the zero vector 0, normalize it by setting $w_i = \frac{{w_i}}{{\lVert w_i \rVert}}$.

To implement the Gram-Schmidt process in Python, you can use the provided function `gram_schmidt_np`, which takes a list of length n containing n NumPy arrays, {vi}, each of length n. The function applies the Gram-Schmidt process and returns the resulting orthonormal set of vectors {wi} in the order they were computed.

To ensure the input is valid, the function checks that the list V contains n arrays of the same length n. If the input is invalid, a ValueError is raised.

In the normalization step, you can check if the norm of wi is less than a small value $\epsilon = 1.19e-7$ instead of checking if it is exactly 0. In such cases, you can replace it with the zero vector instead of normalizing it. The value $\epsilon$, known as "machine-epsilon," can be obtained in NumPy for single precision floating-point numbers using `np.finfo('float32').eps`.


