from collections import namedtuple

from numba import njit

PartialModel = namedtuple("PartialModel", ["users", "items", "user_matrix", "item_matrix"])


@njit
def inplace_axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]
