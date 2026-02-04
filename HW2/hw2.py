import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt
#%% md
## Please complete the solvers in `solver.py`
#%%
import sys

sys.path.append('./')
from solvers import *
#%% md
## Problem 3: Compressive Sensing


#%%
# create the data
np.random.seed(123)
m = 100  # number of measurements
n = 500  # number of variables
k = 10  # number of nonzero variables
s = 0.05  # measurements noise level
#
A_cs = np.random.randn(m, n)
x_cs = np.zeros(n)
x_cs[np.random.choice(range(n), k, replace=False)] = np.random.choice([-1.0, 1.0], k)
b_cs = A_cs.dot(x_cs) + s * np.random.randn(m)
#
lam_cs = 0.1 * norm(A_cs.T.dot(b_cs), np.inf)
#%%
# define the function, prox and the beta constant
def func_f_cs(x):
    f = 0.5 * (LA.norm(A_cs @ x - b_cs, 2)) ** 2
    return f


def func_g_cs(x):
    g = lam_cs * LA.norm(x, 1)
    return g


def grad_f_cs(x):
    grad = A_cs.T @ (A_cs @ x - b_cs)
    return grad


def prox_g_cs(z, t):
    ## z is the gradient step
    ## t = 1/ beta
    x_kp1 = np.zeros_like(z)
    for i in range(len(z)):
        if z[i] > t * lam_cs:
            x_kp1[i] = z[i] - t * lam_cs
        elif np.abs(z[i]) <= t * lam_cs:
            x_kp1[i] = 0
        else:
            x_kp1[i] = t * lam_cs + z[i]

    return x_kp1


# TODO: complete the prox of 1 norm

##==GRADED==##
# TODO: what is the beta value for the smooth part
beta_f_cs = LA.norm(A_cs.T @ A_cs, 2)
#%%
##==GRADED==##
cs_test_dot = np.ones(n)

# Should be a number
func_cs_test = func_f_cs(cs_test_dot) + func_g_cs(cs_test_dot)

# Should be a numpy vector of shape (n, )
grad_f_cs_test = grad_f_cs(cs_test_dot)

# Should be a numpy vector of shape (n, )
prox_g_cs_test = prox_g_cs(cs_test_dot, 1)
#%% md
### Proximal gradient descent on compressive sensing
#%%
# apply the proximal gradient descent solver
x0_cs_pgd = np.zeros(x_cs.size)
x_cs_pgd, obj_his_cs_pgd, err_his_cs_pgd, exit_flag_cs_pgd = optimizeWithPGD(x0_cs_pgd, func_f_cs, func_g_cs, grad_f_cs,
                                                                             prox_g_cs, beta_f_cs)
#%%
# plot signal result
plt.plot(x_cs)
plt.plot(x_cs_pgd, '.')
plt.legend(['true signal', 'recovered'])
plt.title('Compressive Sensing Signal')
plt.show()
#%%
# plot result
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(obj_his_cs_pgd)
ax[0].set_title('function value')
ax[1].semilogy(err_his_cs_pgd)
ax[1].set_title('optimality condition')
fig.suptitle('Proximal Gradient Descent on Compressive Sensing')
plt.show()
#%% md
### Accelerate proximal gradient descent on compressive sensing
# apply the proximal gradient descent solver
x0_cs_apgd = np.zeros(x_cs.size)
x_cs_apgd, obj_his_cs_apgd, err_his_cs_apgd, exit_flag_cs_apgd = \
    optimizeWithAPGD(x0_cs_apgd, func_f_cs, func_g_cs, grad_f_cs, prox_g_cs, beta_f_cs)
#%%
# plot signal result
plt.plot(x_cs)
plt.plot(x_cs_pgd, '.')
plt.legend(['true signal', 'recovered'])
plt.title('Compressive Sensing Signal')
plt.show()
#%%
# plot result
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(obj_his_cs_apgd)
ax[0].set_title('function value')
ax[1].semilogy(err_his_cs_apgd)
ax[1].set_title('optimality condition')
fig.suptitle('Accelerated Proximal Gradient Descent on Compressive Sensing')
plt.show()