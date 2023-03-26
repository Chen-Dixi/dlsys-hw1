import sys
sys.path.append('./python')
sys.path.append('./apps')
from simple_ml import *

import numpy as np
import needle as ndl

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    # print(numerical_grads)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    # print("--------")
    # print(computed_grads)
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    print("numerical_grads")
    print(numerical_grads)
    print("computed_grads")
    print(computed_grads)
    assert error < tol
    return computed_grads

# computed_grads = gradient_check(ndl.divide_scalar, ndl.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))
# computed_grads = gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(5, 4)), ndl.Tensor(np.random.randn(4, 5)))
  
# gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(6, 6, 5, 4)), ndl.Tensor(np.random.randn(4, 3)))
# gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=(0,))
# gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=(0,1))
# gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=1)
# gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4,1)), axes=(0,1))
# gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(3, 3))
# gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1,)), shape=(3, 3, 3))
# gradient_check(lambda A,B,C : ndl.summation((A@B+C)*(A@B), axes=None),
#                    ndl.Tensor(np.random.randn(10,9)),
#                    ndl.Tensor(np.random.randn(9,8)),
#                    ndl.Tensor(np.random.randn(10,8)), backward=True)
# gradient_check(lambda A,B : ndl.summation(ndl.broadcast_to(A,shape=(10,9))*B, axes=None),
#                 ndl.Tensor(np.random.randn(10,1)),
#                 ndl.Tensor(np.random.randn(10,9)), backward=True)
# gradient_check(lambda A,B,C : ndl.summation(ndl.reshape(A,shape=(10,10))@B/5+C, axes=None),
#                 ndl.Tensor(np.random.randn(100)),
#                 ndl.Tensor(np.random.randn(10,5)),
#                    ndl.Tensor(np.random.randn(10,5)), backward=True)
# gradient_check(ndl.log, ndl.Tensor(np.array([2.0, 1.0])))

gradient_check(ndl.relu, ndl.Tensor(np.array([2.0, -1.0])))
gradient_check(ndl.relu, ndl.Tensor(np.array([2.0, 4.0])))

# a = ndl.Tensor(np.array([[2.0, -1.0],[4.2, .3]]))
