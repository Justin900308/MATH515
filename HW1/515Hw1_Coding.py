# require numpy module
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.linalg as la
from numpy import linalg as LA
from matplotlib.patches import Ellipse
import cvxpy as cp
import matplotlib
# import supplementary functions for Homework 1
import sys

sys.path.insert(0, './')
from hw1_supp import *
# import matplotlib

# matplotlib.use('AGG')
# Have to manually set barebones backend or else autograder is mad
# You can change this seed if you want to test that your program works with different random data,
# but please set it back to 123 when you submit your work.
seed = 123


def optimizeWithGD(x0, func, grad, step_size=None, tol=1e-6, max_iter=1000, use_line_search=True):
 """
 Optimize with Gradient Descent

 input
 -----
 x0 : array_like
     Starting point for the solver.
 func : function
     Takes x and return the function value.
 grad : function
     Takes x and returns the gradient of "func".
 step_size : float or None
     If it is a float number and `use_line_search=False`, it will be used as the step size.
     Otherwise, line search will be used
 tol : float, optional
     Gradient tolerance for terminating the solver.
 max_iter : int, optional
     Maximum number of iterations for terminating the solver.
 use_line_search : bool, optional
     When it is true a line search will be used, otherwise `step_size` has to be provided.

 output
 ------
 x : array_like
     Final solution
 obj_his : array_like
     Objective function's values convergence history
 err_his : array_like
     Norm of gradient convergence history
 exit_flag : int
     0, norm of gradient below `tol`
     1, exceed maximum number of iteration
     2, line search fail
     3, other
 """
 # safeguard
 if not use_line_search and step_size is None:
  print('Please specify the step_size or use the line search.')
  return x0, np.array([]), np.array([]), 3

 # initial step
 x = np.copy(x0)
 g = grad(x)
 #
 obj = func(x)
 err = norm(g)
 #
 obj_his = np.zeros(max_iter + 1)
 err_his = np.zeros(max_iter + 1)
 #
 obj_his[0] = obj
 err_his[0] = err

 # start iterations
 iter_count = 0
 while err >= tol:
  if use_line_search:
   step_size = lineSearch(x, g, g, func)
  #
  # if line search fail step_size will be None
  if step_size is None:
   print('Gradient descent line search fail.')
   return x, obj_his[:iter_count + 1], err_his[:iter_count + 1], 2
  #
  # gradient descent step
  #####
  # TODO: with given step_size, complete gradient descent step
  x = x - step_size * g
  #####
  #
  # update function and gradient
  g = grad(x)
  #
  obj = func(x)
  err = norm(g)
  #
  iter_count += 1
  obj_his[iter_count] = obj
  err_his[iter_count] = err
  #
  # check if exceed maximum number of iteration
  if iter_count >= max_iter:
   print('Gradient descent reach maximum number of iteration.')
   return x, obj_his[:iter_count + 1], err_his[:iter_count + 1], 1
 #
 return x, obj_his[:iter_count + 1], err_his[:iter_count + 1], 0


# create b
b = np.array([1.0, 2.0, 3.0])


# define test function
def test_func(x):
    return 0.5 * sum((x - b) ** 2)


# define test gradient
def test_grad(x):
    return x - b


# define test Hessian
def test_hess(x):
    return np.eye(b.size)

# test gradient descent
x0_gd = np.zeros(b.size)
#
x_gd, obj_his_gd, err_his_gd, exit_flag_gd = optimizeWithGD(x0_gd, test_func, test_grad, 1.0)
# check if the solution is correct
err_gd = norm(x_gd - b)
#
print(x_gd, obj_his_gd[-1])
if err_gd < 1e-6:
    print('Gradient Descent: OK')
else:
    print('Gradient Descent: Err')

# fix a random seed
np.random.seed(seed)
# set dimensions and create some random data
m_lgt = 500
n_lgt = 50
A_lgt = 0.3 * np.random.randn(m_lgt, n_lgt)
x_lgt = np.random.randn(n_lgt)
b_lgt = sampleLGT(x_lgt, A_lgt)
lam_lgt = 0.1
# implement logistic function, gradient and Hessian
def lgt_func(x):
    #####
    # TODO: complete the function
    f = lam_lgt / 2 * LA.norm(x, 2) ** 2
    for i in range(m_lgt):
        a_i = A_lgt[i]
        b_i = b_lgt[i]
        f += np.log(1 + np.exp(a_i.T @ x))  ## the log term
        f -= b_i * a_i.T @ x  ## the inner product term
    #####
    return f


#
def lgt_grad(x):
    #####
    # TODO: complete the gradient
    Ax = A_lgt @ x
    grad = -A_lgt.T @ b_lgt
    sigmas = np.zeros(len(Ax))
    for i in range(len(Ax)):
        sigmas[i] = np.exp(Ax[i]) / (1 + np.exp(Ax[i]))
    grad += A_lgt.T @ sigmas
    grad += lam_lgt * x
    #####
    return grad


#
def lgt_hess(x):
    #####
    # TODO: complete the Hessian
    #####
    return None
x0_lgt_gd = np.zeros(n_lgt)
#==GRADED==#
# No need to change anything in this cell.
x_lgt_gd, obj_his_lgt_gd, err_his_lgt_gd, exit_flag_lgt_gd = \
    optimizeWithGD(x0_lgt_gd, lgt_func, lgt_grad, None)

# plot result
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(obj_his_lgt_gd)
ax[0].set_title('function value')
ax[1].semilogy(err_his_lgt_gd)
ax[1].set_title('norm of the gradient')
fig.suptitle('Gradient Descent on Logistic Regression')
plt.show()