"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    """Return the total number of elements described by a tensor shape.

    Args:
        shape: A tuple or list of integers describing tensor dimensions.

    Returns:
        int: The product of all dimensions. An empty shape (a scalar
        tensor) has no dimensions to multiply, so the result is 1 —
        the multiplicative identity.
    """
    total = 1
    for dim in shape:
        total *= dim
    return total

# Step 2 - argsort
def argsort(values):
    """Return the indices that would sort `values` in ascending order.

    Args:
        values: A sequence of comparable elements (e.g. list or tuple).

    Returns:
        list[int]: Positions p such that [values[p] for p in result]
        is `values` sorted smallest to largest. The sort is stable:
        when elements compare equal, the one appearing earlier in the
        input keeps the smaller index, preserving original relative
        order.
    """
    return sorted(range(len(values)), key=values.__getitem__)

# Step 3 - make_op_enums
from enum import Enum


def make_op_enums():
    """Create the four enum classes that name every supported operation.

    Returns:
        tuple[Enum, Enum, Enum, Enum]: (UnaryOps, BinaryOps, ReduceOps,
        MovementOps), each an `enum.Enum` whose members are exactly the
        supported operation kinds for that category.
    """
    unary_ops = Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
    binary_ops = Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    reduce_ops = Enum('ReduceOps', ['SUM', 'MAX'])
    movement_ops = Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])
    return unary_ops, binary_ops, reduce_ops, movement_ops

# Step 4 - LazyBuffer
import numpy as np


class LazyBuffer:
    """Wraps data as an ndarray and exposes shape/dtype for introspection.

    The raw ndarray lives on `self._np`; everything else in the engine
    reads `self.shape` and `self.dtype` and never touches the raw array
    directly.
    """

    def __init__(self, np_array):
        # Convert lists, tuples, scalars, ... into an ndarray. `asarray`
        # does not copy data that is already an ndarray, so wrapping an
        # existing array keeps it as the single shared source of truth.
        self._np = np.asarray(np_array)
        # Record the metadata up front so callers can introspect a
        # buffer (shape, dtype) without reaching into _np.
        self.shape = self._np.shape
        self.dtype = self._np.dtype

# Step 5 - lazybuffer_const
def const(value, shape):
    """Create a new LazyBuffer of `shape` filled entirely with `value`.

    The backing array is always float32 so the result plugs directly
    into the rest of the lazy backend. This is the workhorse used to
    materialize zeros, ones, and broadcast scalars during the backward
    pass.

    Args:
        value: The constant to fill with (int or float, cast to float32).
        shape: Requested shape — a tuple of ints, a single int, or ()
        for a scalar buffer.

    Returns:
        LazyBuffer: Fresh buffer with shape == shape and float32 dtype.
    """
    return LazyBuffer(np.full(shape, value, dtype=np.float32))


LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    """Return a LazyBuffer of uniform random floats in the interval [0, 1).

    Args:
        shape: Requested shape — a tuple of ints, a single int for 1-D,
            or () for a scalar buffer.
        seed: Optional seed. Passing the same seed makes the output fully
            reproducible (identical values on every call); different seeds
            — or None — give independent randomness.

    Returns:
        LazyBuffer: Buffer whose data is float32, with the requested shape.
    """
    rng = np.random.default_rng(seed)
    return LazyBuffer(rng.random(shape, dtype=np.float32))

# Step 7 - lazybuffer_unary_e
_UNARY_FNS = {
    'NEG': np.negative,
    'RELU': lambda x: np.maximum(x, 0),
    'LOG': np.log,
    'EXP': np.exp,
    'SQRT': np.sqrt,
    'SIGMOID': lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def e(self, op):
    """Apply a unary elementwise op and return the result as a new buffer.

    Args:
        op: A UnaryOps enum member (NEG, RELU, LOG, EXP, SQRT, SIGMOID).

    Returns:
        LazyBuffer: Brand-new buffer holding the elementwise result.
        The original buffer and its underlying array are unchanged.
    """
    try:
        fn = _UNARY_FNS[op.name]
    except (AttributeError, KeyError):
        raise ValueError(f"unsupported unary op: {op!r}") from None
    return LazyBuffer(fn(self._np))


LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e (not yet solved)
# TODO: implement

# Step 9 - lazybuffer_r (not yet solved)
# TODO: implement

# Step 10 - lazybuffer_reshape (not yet solved)
# TODO: implement

# Step 11 - lazybuffer_expand (not yet solved)
# TODO: implement

# Step 12 - lazybuffer_permute (not yet solved)
# TODO: implement

# Step 13 - Function (not yet solved)
# TODO: implement

# Step 14 - function_forward_backward_stubs (not yet solved)
# TODO: implement

# Step 15 - apply (not yet solved)
# TODO: implement

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

