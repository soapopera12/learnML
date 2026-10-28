# ML Maths

## Vector maths

### Vectors and Scalars

1. Scalar: single number (e.g., temperature, weight)
2. Vector: list of numbers (has magnitude and direction)

$$
\mathbf{x} = [x_1, x_2, x_3]
$$

---

### Vector Operations (Addition, Dot Product, Norms)

Addition:
$$
\mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2]
$$

Dot Product (similarity):
$$
\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i
$$

Norm (length of vector):
$$
||\mathbf{x}|| = \sqrt{\sum x_i^2}
$$

---

### Matrices and Matrix Operations

Matrix = collection of vectors

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

Matrix multiplication:
$$
AB \neq BA
$$

---

### Matrix Transformation (Geometric Interpretation)

Matrix = transformation of space

$$
\mathbf{y} = A\mathbf{x}
$$

Effect:
1. Rotation
2. Scaling
3. Shearing

---

### Linear Independence and Basis

1. Linearly independent: No vector can be written using others.
2. Basis: Minimum set of vectors to represent space

Understanding for a 2D plane/space:

Two vectors that are not pointing in the same or opposite direction are considered linearly independent and can form a basis for the 2D plane. These two vectors can be scaled and combined to reach any point/vector in the 2D plane. 

$$t = c_1v_1 + c_2v_2$$

$$e.g. v_1 = [2, 1], v_2 = [-1, 1] $$

For vectors $v_1$ and $v_2$ to form the basis their determinant must be non-zero.

$$
\det(\begin{bmatrix} 2 & -1 \\
1 & 1 \end{bmatrix}) = 2(1) - (-1)(1) = 2 + 1 = 3 \neq 0
$$

So they form a basis for the 2D plane. To reach a/any specific point like (5, 4)

t = [5, 4]

$$
c_1 \begin{bmatrix} 2 \\ 1 \end{bmatrix} + c_2 \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 4 \end{bmatrix}
$$

$$
2c_1 - c_2 = 5
c_1 + c_2 = 4
$$

On solving $c_1 = 3$ and $c_2 = 1$

$$
3 \begin{bmatrix} 2 \\ 1 \end{bmatrix} + 1 \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 4 \end{bmatrix}
$$

Just like this the basis vector can reach any other point in the 2D plane.

---

### Rank of a Matrix

Rank = number of independent rows/columns

$$
\text{Rank}(A) = \text{number of independent vectors}
$$

Tells:
1. Information content
2. Redundancy in data

---

### Eigenvalues and Eigenvectors

A matrix transforms vectors

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

1. Eigenvector = vectors remain unchanged in direction after transformation (only scale). 
2. Eigenvalue = scaling factor for the eigen vectors

$$
A\mathbf{v} = \lambda \mathbf{v} 
$$

$$
(A - \lambda I)\mathbf{v} = 0
$$

Here I is the identity matrix of the same size. For a non-zero eigenvector v tp exist, the determinant of (A - λI) must be zero. This is known as the characteristic equation.

For example, for a 2x2 matrix A:

$$
A = \begin{bmatrix} 2 & 1 \\
1 & 2\end{bmatrix}
$$

$$
A - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\
1 & 2 - \lambda \end{bmatrix}
$$

$$
det(A - \lambda I) = (2 - \lambda)(2 - \lambda) - 1 = \lambda^2 - 4\lambda + 3 = 0
$$

On solving this quadratic equation, we get the eigenvalues $\lambda_1 = 1$ and $\lambda_2 = 3$.

For the Eigen vectors we must solve the equation (A - λI)v = 0 for each eigenvalue.

1. For $\lambda$ = 1:

    Substituting $\lambda$ = 1 

    $$
    \begin{bmatrix} 2 - 1 & 1 \\ 1 & 2 - 1 \end{bmatrix} \begin{bmatrix}x_1 \\ x_2 \end{bmatrix} = 0
    $$

    The matrix translates to:
    1. 1x + 1y = 0 => x + y = 0 
    2. 1x + 1y = 0 => x + y = 0

    Both equations simplify to y = -x. We can choose x = 1, then y = -1. So the eigenvector corresponding to $\lambda$ = 1 is $v_1$ = [1, -1].

2. For $\lambda$ = 3:

    Similary we get $v_2$ = [1, 1] as the eigenvector corresponding to $\lambda$ = 3.

---

### Idea of eigen vectors and eigen values

Basically it says that when we transform a vector using a matrix, some vectors only get either scaled up or down, which can be written as

$$
Av = \lambda v
$$

Here $\lambda$ is the scaling factor and v is the eigen vector. A is the transformation matrix.

This can be written as

$$
Av - \lambda v = 0
$$

$$
(A - \lambda I)v = 0
$$

So now if $det(A - \lambda I) = 0$, then it means that there exists eigen vectors for matrix A. 
$A - \lambda I$ squishes space for a vector in a way that it is sent to the origin. Meaning the matrix needs to squish space/transform the vector such that it is sent to the origin. So why determinant? 

**Because determinant is the actual volume of the space that is being transformed. So if the determinant is 0, it means that the space is squished to a lower dimension and hence the vector is sent to the origin.**

---

### Covariance Matrix

Shows how features vary together

$$
\text{Cov}(X, Y) = \frac{1}{n} \sum (x - \bar{x})(y - \bar{y})
$$

Matrix form:
$$
\Sigma = \text{covariance matrix}
$$

---

## Sine and cosine

In math and (ML math) sin and cosine is in radians (in $\pi$ notation), not degrees. 

$$180^{\circ} = \pi \ \text{radians} \approx 3.1416$$
$$5 \ \text{radians} \approx 286.5^{\circ}$$

To convert from radians to degrees use this formula:-

$$\text{degrees} = \text{radians} \times \frac{180}{\pi}$$

So sin($5^{\circ}$) is not sin(5). same goes for cosine

<p align="center">
    <img src="pics/unit-circle.png" alt="normal distro" width="300"><br>
    <em>unit circle for sin/cos understanding</em>
</p>

---

### sine

To calculate sin from radians use the formula

$$
sin(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n+1}}{(2n+1)!}
$$

Which is the taylor series definition of sine. It is essential that x is in radians.

$$sin(\pi/6) → \text{converges to} \ \frac{1}{2}$$
$$sin(5) → \text{converges to} \ -0.9589$$
$$sin(1) → \text{converges to} \ 0.84$$

Sine comes from the unit circle

1. Take a circle of radius 1
2. Start at angle 0
3. Rotate counter clockwise by angle $\theta$
4. sin($\theta$) = y-coordinate of the point
5. so sin($\theta$) = vertical height of the point on the circle

So eg:- If we start at angle $\theta$ and rotate counter-clockwise till $\pi/6$ we get sin($\pi/6$) = 1/2. Which is the y-coordinate of the point on the unit circle. 

There is one more thing about sine which is it's wave representation. As we already know that sin is the y-coordinate of a unit circle that how can it be represented in a wave. Once you unwrap the circle in a linear fashion it creates a wave shape and since sine is periodic the wave becomes periodic too.

<p align="center">
    <img src="pics/unitcircleMBa.jpg" alt="unit_circle" width="500"><br>
    <em>unwrapping the sin for wave representation</em>
</p>

---

### cosine

To calculate cos from radians use the formula

$$
\cos(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n}}{(2n)!}
$$

Which is the Taylor series definition of cosine. It is essential that $x$ is in radians.

$$\cos(\pi/6) \rightarrow \text{converges to} \ \frac{\sqrt{3}}{2}$$
$$\cos(5) \rightarrow \text{converges to} \ 0.2837$$
$$\cos(1) \rightarrow \text{converges to} \ 0.5403$$

Cosine comes from the unit circle

1. Take a circle of radius 1
2. Start at angle 0
3. Rotate counter clockwise by angle $\theta$
4. cos($\theta$) = x-coordinate of the point
5. so cos($\theta$) = horizontal distance of the point from the origin

So eg:- If we start at angle $\theta$ and rotate counter-clockwise till $\pi/6$ we get cos($\pi/6$) = $\sqrt{3}/2$. Which is the x-coordinate of the point on the unit circle. 



