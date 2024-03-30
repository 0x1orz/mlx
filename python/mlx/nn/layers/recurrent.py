from typing import Callable, Optional, override
import mlx.core as mx
import numpy as np
from mlx.nn.layers.activations import tanh
from mlx.nn.layers import Linear
from mlx.nn.layers.base import Module

class RNNBase(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.hh_proj: Module = None

        self.h_0 : mx.array = mx.zeros((1, hidden_size))
        self.h_n : mx.array = None
        self.hh_proj : Module = None
    
    def _extra_repr(self):
        return (
            f"\ninput_dims={self.hh_proj.weight.shape}, hidden_size={self.hh_proj.weight.shape}, bias={"bias" in self.ih_proj},"
        )
    
    @override
    def _cell_fun(self, hh_proj, x, states):
        raise NotImplementedError
    
class ElmanRNN(RNNBase):
    r"""An Elman recurrent layer.

    The input is a sequence of shape ``NLD`` or ``LD`` where:

    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``D`` is the input's feature dimension

    Concretely, for each element along the sequence length axis, this
    layer applies the function:

    .. math::

        h_{t + 1} = \text{tanh} (W_{ih}x_t + W_{hh}h_t + b)

    The hidden state :math:`h` has shape ``NH`` or ``H``, depending on
    whether the input is batched or not. Returns the hidden state at each
    time step, of shape ``NLH`` or ``LH``.

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool, optional): Whether to use a bias. Default: ``True``.
        nonlinearity (callable, optional): Non-linearity to use. If ``None``,
            then func:`tanh` is used. Default: ``None``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Optional[Callable] = None,
    ):
        super().__init__(input_size, hidden_size, bias)

        self.ih_proj = Linear(input_size, hidden_size, bias=bias)
        self.hh_proj = Linear(hidden_size, hidden_size, bias=bias)

        self.nonlinearity = nonlinearity or tanh
        if not callable(self.nonlinearity):
            raise ValueError(
                f"Nonlinearity must be callable. Current value: {nonlinearity}."
            )

    def _cell_fun(self, x, states):
        h = states + x
        h = self.nonlinearity(h)
        return h

    def __call__(self, x, hidden=None):
        self.h_0 = hidden or self.h_0
        all_hidden = []

        batch_size, seq_len = x.shape[:2]
        x = self.ih_proj(x)
        ih = mx.expand_dims(self.h_0, 0)
        for idx in range(seq_len):
            ix = x[..., idx, :]
            ih = self.hh_proj(ih)
            ih = self._cell_fun(ix, ih)
            all_hidden.append(ih)
        self.h_n = ih

        return mx.stack(all_hidden).reshape((seq_len, batch_size, -1)), self.h_n

class GRU(RNNBase):
    r"""A gated recurrent unit (GRU) RNN layer.

    The input has shape ``NLD`` or ``LD`` where:

    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, this layer computes:

    .. math::

        \begin{aligned}
        r_t &= \sigma (W_{xr}x_t + W_{hr}h_t + b_{r}) \\
        z_t &= \sigma (W_{xz}x_t + W_{hz}h_t + b_{z}) \\
        n_t &= \text{tanh}(W_{xn}x_t + b_{n} + r_t \odot (W_{hn}h_t + b_{hn})) \\
        h_{t + 1} &= (1 - z_t) \odot n_t + z_t \odot h_t
        \end{aligned}

    The hidden state :math:`h` has shape ``NH`` or ``H`` depending on
    whether the input is batched or not. Returns the hidden state at each
    time step of shape ``NLH`` or ``LH``.

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool): Whether to use biases or not. Default: ``True``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__(input_size, hidden_size, bias)

        self.ih_proj = Linear(input_size, 3 * hidden_size, bias=bias)
        self.hh_proj = Linear(hidden_size, 3 * hidden_size, bias=bias)

    
    def _cell_fun(self,  x, states):
        x_rz, x_n = x[..., :-self.hidden_size], x[..., -self.hidden_size:]
        h_rz, h_n = states[..., :-self.hidden_size], states[..., -self.hidden_size:]
        rz = mx.sigmoid(x_rz + h_rz)
        r, z= mx.split(rz, 2, axis=-1)
        n = mx.tanh(x_n + r*h_n)
        h = (1-z)*n + z*h_n
        return h
        
    def __call__(self, x, hidden=None):
        self.h_0 = hidden or self.h_0
        all_hidden = []

        batch_size, seq_len = x.shape[:2]

        x = self.ih_proj(x)
        ih = mx.expand_dims(self.h_0, 0)
        for idx in range(seq_len):
            ix = x[..., idx, :]
            ih = self.hh_proj(ih)
            ih = self._cell_fun(ix, ih)
            all_hidden.append(ih)
        self.h_n = ih

        return mx.stack(all_hidden).reshape((seq_len, batch_size, -1)), self.h_n
    

class LSTM(RNNBase):
    r"""An LSTM recurrent layer.

    The input has shape ``NLD`` or ``LD`` where:

    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``D`` is the input's feature dimension

    Concretely, for each element of the sequence, this layer computes:

    .. math::
        \begin{aligned}
        i_t &= \sigma (W_{xi}x_t + W_{hi}h_t + b_{i}) \\
        f_t &= \sigma (W_{xf}x_t + W_{hf}h_t + b_{f}) \\
        g_t &= \text{tanh} (W_{xg}x_t + W_{hg}h_t + b_{g}) \\
        o_t &= \sigma (W_{xo}x_t + W_{ho}h_t + b_{o}) \\
        c_{t + 1} &= f_t \odot c_t + i_t \odot g_t \\
        h_{t + 1} &= o_t \text{tanh}(c_{t + 1})
        \end{aligned}

    The hidden state :math:`h` and cell state :math:`c` have shape ``NH``
    or ``H``, depending on whether the input is batched or not.

    The layer returns two arrays, the hidden state and the cell state at
    each time step, both of shape ``NLH`` or ``LH``.

    Args:
        input_size (int): Dimension of the input, ``D``.
        hidden_size (int): Dimension of the hidden state, ``H``.
        bias (bool): Whether to use biases or not. Default: ``True``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__(input_size, hidden_size, bias)

        self.ih_proj = Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh_proj = Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        self.c_0 = mx.zeros_like(self.h_0)
        self.c_n = None

    def _cell_fun(self, x, states):
        h, c = states
        hx = h + x
        ifo = mx.sigmoid(hx[..., :-self.hidden_size])
        g = mx.tanh(hx[..., -self.hidden_size:])
        i, f, o = mx.split(ifo, 3, axis=-1)
        c = f*c + i*g
        h = o*mx.tanh(c)
        return (h, c)

    def __call__(self, x, hidden=None, cell=None):

        self.h_0 = hidden or self.h_0
        self.c_0 = cell or self.c_0
        all_hidden, all_cell = [], []
        
        ih = self.h_0
        ic = self.c_0

        x = self.ih_proj(x)
        for idx in range(x.shape[-2]):
            ix = x[..., idx, :]
            ih = self.hh_proj(ih)
            ih, ic = self._cell_fun(ix, (ih, ic))
            all_hidden.append(ih)
            all_cell.append(ic)
            
        self.h_n = ih
        self.c_n = ic

        return  mx.stack(all_hidden), mx.stack(all_cell)