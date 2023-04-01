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
    
    assert error < tol
    return computed_grads

#  test_compute_gradient
a = ndl.Tensor(np.random.randn(10,9))
b = ndl.Tensor(np.random.randn(9,8))
c = ndl.Tensor(np.random.randn(10,8))


gradient_check(lambda A,B,C : ndl.summation((A@B+C)*(A@B), axes=None), a, b, c, backward=True)

# tensor 和 tensor.grad 的精度是否匹配
assert a.dtype == a.grad.dtype

# # test forward/backward pass for relu
# np.testing.assert_allclose(ndl.relu(ndl.Tensor([[-46.9 , -48.8 , -45.45, -49.  ],
#        [-49.75, -48.75, -45.8 , -49.25],
#        [-45.65, -45.25, -49.3 , -47.65]])).numpy(), np.array([[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]]))
# gradient_check(ndl.relu, ndl.Tensor(np.random.randn(5,4)))