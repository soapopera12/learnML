# Less Important Stuff

## Derivative (Rate of Change)

A derivative of a function represents the rate at which a function's value changes as its input changes. In simpler terms, it measures how the output of a function responds to small changes in the input.

The formal definition of a derivative is:

$$
f'(x) = \frac{df}{dx} = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}
$$

Also recall the formula for slope:

$$
\text{slope} = \frac{y_2 - y_1}{x_2 - x_1}
$$

Here:
$$
y_1 = f(x), \quad y_2 = f(x + h)
$$

So, the derivative represents the slope of the secant line as the change in input $h$ becomes very small. In the limit as $h \to 0$, this secant line becomes the tangent line to the curve at point $x$.

Once we know the slope (gradient), we can decide how far to move along this direction. This "step size" is controlled by the learning rate in optimization algorithms like gradient descent.

$$
h \rightarrow \text{change in input (very small)}
$$

---

## Integration (Accumulation of Change)

Integration represents the accumulation of quantities over an interval. In simpler terms, it adds up infinitely small pieces to compute a total.

The definite integral is defined as:

$$
\int_a^b f(x)\,dx = \lim_{n \rightarrow \infty} \sum_{i=1}^{n} f(x_i)\,\Delta x
$$

Here:
$$
\Delta x = \frac{b - a}{n}
$$

Here the interval between a and b are infinite therefore n tends to go to infinity in the equation. This means we divide the interval $[a, b]$ into very small parts, evaluate the function at each part, and sum them up.

As $n \to \infty$, the width $\Delta x \to 0$, and the summation becomes an integral.

Each small piece contributes:

$$
\text{small contribution} \approx f(x)\cdot dx
$$

So the total accumulation becomes:

$$
\int_a^b f(x)\,dx
$$

Geometrically, integration represents the area under the curve between $a$ and $b$.

$$
dx \rightarrow \text{a very small width (infinitesimal change in } x\text{)}
$$

---

## Pseudo-Random Number Generators (PRNG)

Computers do not generate truly random numbers. Instead, they use **Pseudo-Random Number Generators (PRNGs)**, which produce sequences that appear random but are generated deterministically.

Most PRNGs follow a recursive relation:

$$ 
x_{n+1} = f(x_n) 
$$

where the next number depends on the current number.

**Linear Congruential Generator (LCG)**

A classic example of a PRNG is the **Linear Congruential Generator**:

$$
x_{n+1} = (a x_n + c) \bmod m
$$

**Where:**
1. $x_n$ is the current number
2. $a, c, m$ are constants
3. $x_{n+1}$ is the next number in the sequence


The sequence starts from an initial value called the **seed**:

$$
\text{seed} \rightarrow x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots
$$

The entire sequence is completely determined by the seed.

**Choosing Parameters ($a, c, m$)**

The constants $a, c, m$ are carefully chosen to ensure good randomness properties such as:
1. Long period (sequence does not repeat quickly)
2. Uniform distribution of values
3. Minimal correlation or patterns


To achieve a **maximum period** of $m$, the following conditions (Hull--Dobell Theorem) must be satisfied:
1. $c$ and $m$ are coprime
2. $a - 1$ is divisible by all prime factors of $m$
3. If $m$ is divisible by $4$, then $a - 1$ must also be divisible by $4$

**Example (commonly used values):**
$$
m = 2^{32}, \quad a = 1664525, \quad c = 1013904223
$$

**Period of the Generator**

A PRNG produces numbers until the sequence repeats. This length is called the **period**.

For an LCG:
$$
\text{Period} \leq m
$$

For example:
$$
m = 2^{32} \Rightarrow \text{maximum period} \approx 4 \text{ billion}
$$

Thus, the sequence:
$$
\text{seed} \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots
$$

continues until it eventually cycles back to the initial value.

---