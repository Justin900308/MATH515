# %% [markdown]
# # AMATH 515 Homework 3
# 
# **Due Date: 02/20/2025 at 23:59 PM PDT**
# 
# *Homework Instruction*: Please follow order of this notebook and fill in the codes where commented as `TODO`. **please submit both** `proxes.py` **and** `515Hw3_Coding.ipynb` to Gradescope. You'll have **10 attempts** to pass the tests.
# %%
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# %% [markdown]
# # Problem 5
# %% [markdown]
# ## Please complete the code in `proxes.py`
# %%
import sys

sys.path.append('./')
from proxes import *
# %% [markdown]
# ## Test cases
# %%
np.random.seed(124)
m = 5
n = 2
k = 1
z = np.random.randn(m, n)
x = np.zeros((m, n))
# %%
for i in range(m):
    x[i] = prox_csimplex(z[i], k)
# %%
#==GRADED==#
print(x)
# %% [markdown]
# If you did everything right then the projections of the points on the plot below should be sitting on the black interval that connects [0,1] and [1,0]
# %%
plt.clf()
plt.arrow(-0.5, 0.0, 2.0, 0.0, head_width=0.1)
plt.arrow(0.0, -0.5, 0.0, 2.0, head_width=0.1)
plt.plot([0.0, 1.0], [1.0, 0.0], '-k')
for i in range(m):
    plt.plot([z[i][0], x[i][0]], [z[i][1], x[i][1]], 'o--')
plt.axis('equal')
# %% [markdown]
# # Problem 6
# %%
# Build a dataset for matrix completion
import numpy as np
import matplotlib.pyplot as plt
from solvers import optimizeWithAPGD
from proxes import prox_csimplex
from proxes import prox_l1
# %%
d = 250
rank = 5
#Only see 8% of the entries
mask_prob = 0.08

gen = np.random.default_rng(seed=515)
original_matrix = gen.standard_normal(size=(d, d))

P = gen.binomial(1, mask_prob, (d, d))

U, sigma, Vt = np.linalg.svd(original_matrix)
sigma[rank:] = 0
X = (U @ np.diag(sigma) @ Vt)
X = X / np.sqrt(np.mean(X ** 2))

X_observed = X * P
# %%
def reconstruction_loss(Y):
    """This is our out of sample reconstruction loss. Only for evaluation!
    """
    return np.mean((Y - X) ** 2)


def l2_loss(Y):
    return np.sum(P * ((Y - X_observed) ** 2)) / 2


def l2_grad(Y):
    return P * (Y - X_observed)  ## element wise product


def nuclear_norm(Y):
    return np.linalg.norm(Y, 'nuc')
# %%
##==GRADED==##

proxes_test_matrix = np.array([
    [1, 3, 5],
    [2, 1, 4],
    [1, 1, 1]
])

nuclear_prox_test = nuclear_prox(proxes_test_matrix, 1.5)
rank_proj_test = rank_project(proxes_test_matrix, 2)
print(np.linalg.matrix_rank(rank_proj_test), rank_proj_test)
print(nuclear_prox_test)
# %%
def apgd_nuclear_norm_completion(Y0, lam, tol=1e-5):
    Y_sol, obj_his, err_his, exit_flag = optimizeWithAPGD(
        Y0,
        l2_loss,
        lambda x: lam * nuclear_norm(x),
        l2_grad,
        lambda x, t: nuclear_prox(x, lam * t),
        beta_f=1,
        max_iter=1000,
        tol=tol
    )

    return Y_sol, obj_his, err_his, exit_flag


def apgd_rank_constrained_completion(Y0, k, tol=1e-5):
    Y_sol, obj_his, err_his, exit_flag = optimizeWithAPGD(
        Y0,
        l2_loss,
        lambda x: 0,
        l2_grad,
        lambda x, t: rank_project(x, k),
        beta_f=1,
        max_iter=1000,
        tol=tol
    )
    return Y_sol, obj_his, err_his, exit_flag
# %% [markdown]
# ## Solve an example
# %%
##==GRADED==##

#Initialize with the observed entries
# Y0 = X_observed.copy()
# lam = 0.2
#
# nuclear_example_sol, obj_his, err_his, exit_flag = apgd_nuclear_norm_completion(Y0, lam)
#
# # example_sol is graded here
# plt.plot(np.log10(err_his))

# %%
import cvxpy as cp

##==GRADED==##
#Initialize with the observed entries
Y0 = X_observed.copy()
lam = 0.2


def cvxpy_solver(X,lam):
    print("starting cvxpy solver")
    Y_sol = cp.Variable((d, d))
    f = cp.norm(cp.multiply(P, (Y_sol - X)), "fro") ** 2 + lam * cp.norm(Y_sol,"nuc")
    prob = cp.Problem(cp.Minimize(f))
    prob.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=5000)
    Y_sol_val = Y_sol.value

    print(prob.value)
    return Y_sol.value


Y_sol = cvxpy_solver(X_observed, lam)
# example_sol is graded here
# plt.plot(np.log10(err_his))
# %%
print("Rank of our recovered matrix:", (np.sum(np.linalg.svd(nuclear_example_sol)[1] > 1e-6)))

#This should be about 0.1
reconstruction_error = reconstruction_loss(nuclear_example_sol)
print("Reconstruction error on the full dataset:", reconstruction_error)

print("Note that the original entrywise variance is 1, \
so in some sense our recovered estimates are about 90% accurate, using only 8% of the data")
# %% [markdown]
# ## Sweep through many values of lambda
# %%
import time

lam_vals = np.append(np.arange(0.25, 3, 0.25)[::-1], np.array([0.1, 0.05, 0.01]))
##==GRADED==##
#Compute the nuclear norm regularized matrix completion for each value of lambda in lam_vals
# (hint: use apgd_nuclear_norm_completion with tol=1e-3)
Y_solution_list = []
for lam in lam_vals:
    time_0 = time.time()
    Y_solution_lam, _, _, _ = apgd_nuclear_norm_completion(Y0, lam, 1e-3)
    time_1 = time.time()
    print("Esp time for ", lam, time_1 - time_0)
    Y_solution_list.append(Y_solution_lam)
ss = 2
# Y_solution_list = [???]

# Y_solution_list should be a python list of the solutions (2d array) for each regularization strength
# Y_solution_list[i] is the solution with the regularization strength lam_vals[i]
# %%
## Graded ##
nuc_parameter_sweep_solutions = np.array(Y_solution_list)
# %%
plt.plot(lam_vals, [reconstruction_loss(y) for y in Y_solution_list])
plt.ylabel("Reconstruction error")
plt.xlabel("Regularization strength lambda")
plt.title("Overall Reconstruction Error")
# %% [markdown]
# You should find that as lambda goes to 0, the reconstruction error goes down. This is because we begin to approximate a certain semidefinite programming problem, minimizing nuclear norm subject to the observed entries being exactly correct. This problem has very good theoretical guarantees, including exact recovery under certain conditions!
# 
# See this Candes and Tao paper if you're interested: https://arxiv.org/abs/0903.1476
# %% [markdown]
# #### Just because the set of matrices with constrained rank isn't convex, doesn't mean that we can't try accelerated projected gradient descent on it! We warm start with the last solution generated from nuclear norm regularization. 
# %%
Y_rank5, obj_his, err_his, exit_flag = apgd_rank_constrained_completion(Y_solution_list[-1], 5)

plt.plot(np.log10(err_his))
nonconvex_reconstruction = reconstruction_loss(Y_rank5)
print("Rank 5 reconstruction loss:", nonconvex_reconstruction)
# %% [markdown]
# You should find that we have exact recovery of the original matrix X by using this non-convex constraint despite only observing 8% of the entries. I find this **truly remarkable!**