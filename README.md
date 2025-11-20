# Polynomial Curve Fitting (5th degree) — Gradient Descent (Python)

## Overview
This project implements a 5th degree polynomial curve-fitting algorithm using gradient descent.  
Given seven y-values sampled at x ∈ {−3, −2, −1, 0, 1, 2, 3}, the script attempts to learn coefficients k_5, k_4, ..., k_1 and constant k_0 for the polynomial:

p(x) = k_5 * x^5 + k_4 * x^4 + k_3 * x^3 + k_2 * x^2 + k_1 * x + k_0

The code is implemented in pure Python (no third-party libs required, uses only Python’s built-in modules).

## Details
- Random initialization of coefficients.
- Domain normalization to improve numerical stability.
- Mean squared error loss.
- Manual gradient computation and gradient-descent updates.
- Simple Command Line Interface input of seven y-values and number of iterations.

## Limitations & Notes
- Educational implementation: intentionally minimal to demonstrate gradient descent.
- Convergence may require many iterations and will depend on initialization, learning rate, and the actual function's complexity.
- Not optimized for production: no analytic least-squares, no optimizer (Adam), no regularization, no batching.
- Results vary between runs due to random initialization.

## Usage
```bash
python src/polynomial_fit.py
