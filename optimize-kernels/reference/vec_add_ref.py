import numpy as np


def reference(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int):
    # c is preallocated output; return it for clarity
    np.copyto(c, a + b)
    return c
