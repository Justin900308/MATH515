# %% [markdown]
# # AMATH 515 Homework 4 Coding Assignment
# **Name:** <br>
# **Student Number:** <br>
# **Due: 3/13/2026**
# %%
import sys
sys.path.append("./")
import dataUtil

import numpy as np
import scipy.sparse as sp
from solvers import optimizeWithIP
# %% [markdown]
# ## Problem 5: Interior Point Method
# Implement an interior point method in `solvers.py` to solve the problem 
# $$ \min_x \frac{1}{2}\|Ax-b\|^2 \quad \text{s.t.} \quad Cx \leq d. $$
# Test your algorithm using the box constrained problem below.
# %%
np.random.seed(123)
A = np.random.randn(10, 5)
b = np.random.randn(10)
C = np.vstack([np.eye(5), -np.eye(5)])
d = np.ones(10)
x0 = np.zeros(5)

## GRADED ##
x_ip, obj_his, err_his, flag = optimizeWithIP(x0, A, b, C, d)
p4_err = err_his[-1] if len(err_his) > 0 else 1.0
print("Optimal x value:", x_ip)
print("Objective values:", obj_his)
print("Final error IP:", p4_err)
# # %% [markdown]
# # ## Problem 6: Nonconvex Matrix Completion via PALM
# #
# # In this problem, we apply the Proximal Alternating Linearized Minimization (PALM) algorithm to the Movielens dataset. We aim to find low-rank matrices that approximate the user-movie ratings by solving the following nonconvex optimization problem:
# #
# # $$\min_{U,V} \frac{1}{2}\|b - \mathcal{A}(U^TV)\|^2  + \frac{\lambda}{2}\|U\|_F^2 + \frac{\lambda}{2}\|V\|_F^2$$
# #
# # **Notation:**
# # * $m$: Total number of users
# # * $n$: Total number of movies
# # * $k$: Rank of the factorization (latent features)
# # * $\lambda$: Regularization parameter for the Frobenius norm penalties
# # * $U \in \mathbb{R}^{k \times m}$: User feature matrix
# # * $V \in \mathbb{R}^{k \times n}$: Movie feature matrix
# # %%
# # Load the Movielens data
# num_measurements, user_ids, movie_ids, ratings = dataUtil.load("./ratings.bin")
# train_indices, test_indices = dataUtil.partition(num_measurements)
#
# # Problem dimensions
# num_users = np.max(user_ids) + 1
# num_movies = np.max(movie_ids) + 1
# rank = 5
# lambda_reg = 0.1
#
# # Initialize U and V with random normal values
# np.random.seed(123)
# U_matrix = np.random.randn(rank, num_users)
# V_matrix = np.random.randn(rank, num_movies)
#
# # Convert training data to Sparse Matrices for instantaneous row/col slicing
# # CSR is fast for extracting User rows. CSC is fast for extracting Movie columns.
# R_csr = sp.csr_matrix(
#     (ratings[train_indices], (user_ids[train_indices], movie_ids[train_indices])),
#     shape=(num_users, num_movies)
# )
# R_csc = R_csr.tocsc()
#
# def compute_objective(U, V, R_csr):
#     """Vectorized objective computation"""
#     obj = 0.5 * lambda_reg * (np.sum(U**2) + np.sum(V**2))
#     res_sq = 0.0
#     for u in range(num_users):
#         start, end = R_csr.indptr[u], R_csr.indptr[u+1]
#         if start == end: continue
#         m_idx = R_csr.indices[start:end]
#
#         preds = np.dot(U[:, u], V[:, m_idx])
#         res_sq += np.sum((preds - R_csr.data[start:end])**2)
#     return obj + 0.5 * res_sq
#
# max_iter = 100
# iteration = 0

# # PALM Iteration
# while iteration < max_iter:
#
#     # Update U (Users)
#     for u in range(num_users):
#         start, end = R_csr.indptr[u], R_csr.indptr[u+1]
#         if start == end: continue # User has no ratings
#
#         m_idx = R_csr.indices[start:end]
#         true_ratings = R_csr.data[start:end]
#
#         V_u = V_matrix[:, m_idx]                     # Shape: (k, num_movies_rated)
#
#         # TODO: Vectorized predictions, residuals, gradient, and Lipschitz constant
#         preds = ??
#         residual = ??
#         grad_u = ??
#         Lu = ??
#
#         # TODO: Proximal gradient step to update U_matrix[:, u]
#         U_matrix[:, u] = ??
#
#     # Update V (Movies)
#     for m in range(num_movies):
#         start, end = R_csc.indptr[m], R_csc.indptr[m+1]
#         if start == end: continue # Movie has no ratings
#
#         u_idx = R_csc.indices[start:end]
#         true_ratings = R_csc.data[start:end]
#
#         U_m = U_matrix[:, u_idx]                     # Shape: (k, num_users_who_rated)
#
#         # TODO: Vectorized predictions, residuals, gradient, and Lipschitz constant
#         preds = ??
#         residual = ??
#         grad_v = ??
#         Lv = ??
#
#         # TODO: Proximal gradient step to update V_matrix[:, m]
#         V_matrix[:, m] = ??
#
#     iteration += 1
#
#     if iteration % 10 == 0:
#         print(f"Iteration {iteration:3d} | Objective: {compute_objective(U_matrix, V_matrix, R_csr):.5e}")
#
# # Test MSE calculation
# # Computes the dot product for all test pairs concurrently
# u_test = user_ids[test_indices]
# m_test = movie_ids[test_indices]
# r_test = ratings[test_indices]
#
# # Element-wise multiply the specific columns of U and V, then sum along the features (k)
# test_preds = np.sum(U_matrix[:, u_test] * V_matrix[:, m_test], axis=0)
# mse = np.mean((test_preds - r_test)**2)
#
# ## GRADED ##
# print(f"Final Test MSE: {mse:.5f}")
# p5_mse_ans = mse
# # %% [markdown]
# # ## Problem 7: Robust Matrix Completion
# #
# # Consider the robustified variant of matrix completion:
# # $$\min_{U,V,s} \frac{1}{2}\|b-\mathcal{A}(U^TV)-s\|^2 +\lambda_2 \|s\|_1.$$
# #
# # Modify your code from the previous problem to solve this factorized variant of robust PCA. *Hint: Apply the Huber derivative logic.*
# #
# # As proven in the written portion, minimizing out the sparse outlier variables $s$ point-wise is equivalent to applying the Huber penalty to the residuals.
# #
# # **Notation:**
# # * $m$: Total number of users
# # * $n$: Total number of movies
# # * $k$: Rank of the factorization (latent features)
# # * $\lambda_1$: Regularization parameter for the Frobenius norm penalties
# # * $\lambda_2$: Threshold parameter for the Huber penalty (robustness)
# # * $U \in \mathbb{R}^{k \times m}$: User feature matrix
# # * $V \in \mathbb{R}^{k \times n}$: Movie feature matrix
# # %%
# # Load the Movielens data & Setup Dimensions
# num_measurements, user_ids, movie_ids, ratings = dataUtil.load("./ratings.bin")
# train_indices, test_indices = dataUtil.partition(num_measurements)
#
# num_users = np.max(user_ids) + 1
# num_movies = np.max(movie_ids) + 1
# rank = 5
# lam1 = 0.1  # Regularization parameter
# lam2 = 1.0  # Huber penalty threshold
#
# # Initialize U and V with random normal values
# np.random.seed(123)
# U_r = np.random.randn(rank, num_users)
# V_r = np.random.randn(rank, num_movies)
#
# # Convert training data to Sparse Matrices for instantaneous row/col slicing
# R_csr = sp.csr_matrix(
#     (ratings[train_indices], (user_ids[train_indices], movie_ids[train_indices])),
#     shape=(num_users, num_movies)
# )
# R_csc = R_csr.tocsc()
#
# def compute_rpca_objective(U, V, R_csr, lam1, lam2):
#     """Vectorized RPCA objective computation with Huber loss"""
#     obj = 0.5 * lam1 * (np.sum(U**2) + np.sum(V**2))
#     huber_sum = 0.0
#     for u in range(num_users):
#         start, end = R_csr.indptr[u], R_csr.indptr[u+1]
#         if start == end: continue
#         m_idx = R_csr.indices[start:end]
#
#         preds = np.dot(U[:, u], V[:, m_idx])
#         r_vals = preds - R_csr.data[start:end]
#
#         # Huber penalty
#         abs_r = np.abs(r_vals)
#         huber_sum += np.sum(np.where(abs_r <= lam2, 0.5 * r_vals**2, lam2 * abs_r - 0.5 * lam2**2))
#     return obj + huber_sum
#
# max_iter = 100
# iteration = 0
#
# # Optimized RPCA PALM Iteration
# while iteration < max_iter:
#
#     # Update U (Users)
#     for u in range(num_users):
#         start, end = R_csr.indptr[u], R_csr.indptr[u+1]
#         if start == end: continue
#
#         m_idx = R_csr.indices[start:end]
#         true_ratings = R_csr.data[start:end]
#
#         V_u = V_r[:, m_idx]                          # Shape: (k, num_movies_rated)
#
#         # TODO: Vectorized predictions and standard residual
#         preds = ??
#         residual = ??
#
#         # TODO: Apply the Huber derivative (clipping the residuals)
#         robust_residual = ??
#
#         # TODO: Calculate robust gradient and Lipschitz constant
#         grad_u = ??
#         Lu = ??
#
#         # TODO: Proximal gradient step, apply (2.0 / Lu) step size
#         U_r[:, u] = ??
#
#     # Update V (Movies)
#     for m in range(num_movies):
#         start, end = R_csc.indptr[m], R_csc.indptr[m+1]
#         if start == end: continue
#
#         u_idx = R_csc.indices[start:end]
#         true_ratings = R_csc.data[start:end]
#
#         U_m = U_r[:, u_idx]                          # Shape: (k, num_users_who_rated)
#
#         # TODO: Vectorized predictions and standard residual
#         preds = ??
#         residual = ??
#
#         # TODO: Apply the Huber derivative (clipping the residuals)
#         robust_residual = ??
#
#         # TODO: Calculate robust gradient and Lipschitz constant
#         grad_v = ??
#         Lv = ??
#
#         # TODO: Proximal gradient step, apply (2.0 / Lv) step size
#         V_r[:, m] = ??
#
#     iteration += 1
#
#     if iteration % 10 == 0:
#         print(f"Iteration {iteration:3d} | Objective: {compute_rpca_objective(U_r, V_r, R_csr, lam1, lam2):.5e}")
#
# # Test MSE calculation
# u_test = user_ids[test_indices]
# m_test = movie_ids[test_indices]
# r_test = ratings[test_indices]
#
# test_preds = np.sum(U_r[:, u_test] * V_r[:, m_test], axis=0)
# mse_r = np.mean((test_preds - r_test)**2)
#
# ## GRADED ##
# print(f"Final Test MSE RPCA: {mse_r:.5f}")
# p6b_mse_ans = mse_r
# # %% [markdown]
# # ## Problem 6(c): Non-Negative Matrix Completion
# #
# # Non-negative matrix factorization makes the additional assumption that all latent features must be non-negative:
# # $$U, V \geq 0.$$
# #
# # This leads to the formulation:
# # $$\min_{U,V}\frac{1}{2}\|b-\mathcal{A}(U^TV)\|^2 + \frac{\lambda_1}{2}\|U\|_F^2+\frac{\lambda_1}{2}\|V\|_F^2 + \delta_{\mathbb{R}_+^{k\times m}}(U)+\delta_{\mathbb{R}_+^{k\times n}}(V)$$
# #
# # Apply your proximal operator for these functions into the PALM implementation to evaluate the error.
# #
# # *Hints:* * *Use `np.maximum(..., 0.0)` to project onto the non-negative orthant.*
# # * *Because this projection can push latent features to exactly $0.0$, be sure to add a tiny epsilon (e.g., `1e-9`) when calculating your Lipschitz constants $L_u$ and $L_v$ to prevent divide-by-zero errors!*
# # %%
# # Load the Movielens data & Setup Dimensions
# num_measurements, user_ids, movie_ids, ratings = dataUtil.load("./ratings.bin")
# train_indices, test_indices = dataUtil.partition(num_measurements)
#
# num_users = np.max(user_ids) + 1
# num_movies = np.max(movie_ids) + 1
# rank = 5
# lam1 = 0.1  # Regularization parameter
# lam2 = 1.0  # Huber penalty threshold
#
# # Initialize U and V with random normal values
# np.random.seed(123)
# U_nn = np.random.randn(rank, num_users)
# V_nn = np.random.randn(rank, num_movies)
#
# # Convert training data to Sparse Matrices for instantaneous row/col slicing
# R_csr = sp.csr_matrix(
#     (ratings[train_indices], (user_ids[train_indices], movie_ids[train_indices])),
#     shape=(num_users, num_movies)
# )
# R_csc = R_csr.tocsc()
#
# def compute_rpca_objective(U, V, R_csr, lam1, lam2):
#     """Vectorized RPCA objective computation with Huber loss"""
#     obj = 0.5 * lam1 * (np.sum(U**2) + np.sum(V**2))
#     huber_sum = 0.0
#     for u in range(num_users):
#         start, end = R_csr.indptr[u], R_csr.indptr[u+1]
#         if start == end: continue
#         m_idx = R_csr.indices[start:end]
#
#         preds = np.dot(U[:, u], V[:, m_idx])
#         r_vals = preds - R_csr.data[start:end]
#
#         abs_r = np.abs(r_vals)
#         huber_sum += np.sum(np.where(abs_r <= lam2, 0.5 * r_vals**2, lam2 * abs_r - 0.5 * lam2**2))
#     return obj + huber_sum
#
# max_iter = 100
# iteration = 0
#
# # Optimized Non-Negative PALM Iteration
# while iteration < max_iter:
#
#     # Update U (Users)
#     for u in range(num_users):
#         start, end = R_csr.indptr[u], R_csr.indptr[u+1]
#         if start == end: continue
#
#         m_idx = R_csr.indices[start:end]
#         true_ratings = R_csr.data[start:end]
#
#         V_u = V_nn[:, m_idx]                         # Shape: (k, num_movies_rated)
#
#         # TODO: Vectorized predictions, residual, robust residual, and gradient
#         preds = ??
#         residual = ??
#         robust_residual = ??
#         grad_u = ??
#
#         # TODO: Calculate Lipschitz constant (add + 1e-9 to prevent divide-by-zero!)
#         Lu = ??
#
#         # TODO: Proximal gradient step with non-negativity constraint, apply (2.0 / Lu) step size
#         U_nn[:, u] = ??
#
#     # Update V (Movies)
#     for m in range(num_movies):
#         start, end = R_csc.indptr[m], R_csc.indptr[m+1]
#         if start == end: continue
#
#         u_idx = R_csc.indices[start:end]
#         true_ratings = R_csc.data[start:end]
#
#         U_m = U_nn[:, u_idx]                         # Shape: (k, num_users_who_rated)
#
#         # TODO: Vectorized predictions, residual, robust residual, and gradient
#         preds = ??
#         residual = ??
#         robust_residual = ??
#         grad_v = ??
#
#         # TODO: Calculate Lipschitz constant (add + 1e-9 to prevent divide-by-zero!)
#         Lv = ??
#
#         # TODO: Proximal gradient step with non-negativity constraint, apply (2.0 / Lv) step size
#         V_nn[:, m] = ??
#
#     iteration += 1
#
#     if iteration % 10 == 0:
#         print(f"Iteration {iteration:3d} | Objective: {compute_rpca_objective(U_nn, V_nn, R_csr, lam1, lam2):.5e}")
#
# # Test MSE calculation
# u_test = user_ids[test_indices]
# m_test = movie_ids[test_indices]
# r_test = ratings[test_indices]
#
# test_preds = np.sum(U_nn[:, u_test] * V_nn[:, m_test], axis=0)
# mse_nn = np.mean((test_preds - r_test)**2)
#
# ## GRADED ##
# print(f"Final Test MSE Non-Negative: {mse_nn:.5f}")
# p6c_mse_ans = mse_nn