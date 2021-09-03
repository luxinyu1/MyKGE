import numpy as np

def convert_to_one_hot(x, depth):
    return np.eye(depth)[x]