# Copyright © 2024 Apple Inc.

import math
from typing import Callable, Optional

import mlx.core as mx
from mlx.nn.layers.activations import tanh
from mlx.nn.layers import Linear
from mlx.nn.layers.base import Module


class RNN(Module):
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
        num_layers: int = 1,
        bidirectional: bool = True,
        bias: bool = True,
        nonlinearity: Optional[Callable] = None,
    ):
        super().__init__()

        self.nonlinearity = nonlinearity or tanh
        if not callable(self.nonlinearity):
            raise ValueError(
                f"Nonlinearity must be callable. Current value: {nonlinearity}."
            )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._ih_proj = Linear(input_dims=input_size, output_dims=hidden_size, bias=bias)

        bidirection = 2 if bidirectional else 1

        assert num_layers >= 1, "num_layers of the hidden layers should be more than 1."
        self.hidden = mx.zeros((bidirection*self.num_layers, hidden_size))
        self._hh_proj = [Linear(input_dims=hidden_size, output_dims=hidden_size, bias=bias) 
                         for _ in range(bidirection*self.num_layers)]

    def _cell_fun(self, hh_proj, x, hidden):
        h = hh_proj(hidden) + x
        h = self.nonlinearity(h)
        return h

    def _extra_repr(self):
        return (
            f"input_dims={self._ih_proj.weight.shape[0]}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, \n\
                bidirectional={len(self._hh_proj)>self.num_layers}"
            f"nonlinearity={self.nonlinearity}, bias={"bias" in self._ih_proj}"
        )

    def __call__(self, x, hidden=None):
        x = self._ih_proj(x)
        self.hidden = hidden or self.hidden
        all_hidden = []
        last_hidden = None
        batch_size, seqlen = x.shape[:2]
        for hh_idx, hh_proj in enumerate(self._hh_proj):
            steps_hidden = []
            ih = self.hidden[hh_idx, :]
            for idx in range(seqlen):

                if hh_idx ==0:
                    ix = x[..., idx, :] 
                elif hh_idx == self.num_layers: 
                    ix = x[..., seqlen-1 - idx, :]
                else:
                    ix = last_hidden[idx]

                ih = self._cell_fun(hh_proj, ix, ih)
                steps_hidden.append(ih)

            if hh_idx == self.num_layers - 1:
                all_hidden.extend(steps_hidden)
            elif hh_idx == 2*self.num_layers - 1:
                all_hidden.extend(steps_hidden[::-1])

            last_hidden = steps_hidden
        return mx.stack(all_hidden).reshape((seqlen, batch_size, -1))

class GRU(Module):
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
        num_layers: int = 1,
        bidirectional: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        bidirection = 2 if bidirectional else 1
        self._ih_proj = Linear(input_size, 3*hidden_size, bias=bias)

        assert num_layers >= 1, "num_layers of the hidden layers should be more than 1."
        self.hidden = mx.zeros(bidirection*self.num_layers, self.hidden_size)
        self._hh_proj = [Linear(hidden_size, 3*hidden_size, bias=bias) 
                         for _ in range(bidirection*self.num_layers)]
   

    def _extra_repr(self):
        return (
            f"input_dims={self._ih_proj.weight.shape[0]}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, \n\
                bidirectional={len(self._hh_proj)>self.num_layers}, bias={"bias" in self._ih_proj}"
        )
    
    def _cell_fun(self, _hh_proj, x, hideen):
        x_rz, x_n = x[..., :-self.hidden_size], x[...,-self.hidden_size:]
        h = _hh_proj(hideen)
        h_rz, h_n = h[..., :-self.hidden_size], h[...,-self.hidden_size:]
        rz = mx.sigmoid(x_rz + h_rz)
        r, z= mx.split(rz, 2, axis=-1)
        n = mx.tanh(x_n + r*h_n)
        h = (1-z)*n + z*h
        return h
        
    def __call__(self, x, hidden=None):
        x = self._ih_proj(x)
        self.hidden = hidden or self.hidden
        all_hidden = []
        last_hidden = None
        batch_size, seqlen = x.shape[:2]
        for hh_idx, hh_proj in enumerate(self._hh_proj):
            ih = self.hidden[hh_idx, :]
            step_hidden = []
            for idx in range(seqlen):
                if hh_idx ==0:
                    ix = x[..., idx, :] 
                elif hh_idx == self.num_layers:
                    ix = x[..., seqlen-1 - idx, :]
                else:
                    ix = last_hidden[idx]

                ih = self._cell_fun(hh_proj, ix, ih)
                step_hidden.append(ih)
                
            if hh_idx == self.num_layers - 1:
                all_hidden.extend(step_hidden)
            elif hh_idx == 2*self.num_layers - 1:
                all_hidden.extend(step_hidden[::-1])

            last_hidden = step_hidden

        return mx.stack(all_hidden).reshape((seqlen, batch_size, -1))
    

class LSTM(Module):
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
        num_layers: int,
        bidirectional: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        bidirection = 2 if bidirectional else 1

        self._ih_proj = Linear(input_size, 4*hidden_size, bias=bias)

        assert num_layers >= 1, "num_layers of the hidden layers should be more than 1."
        self.hidden = mx.zeros(bidirection*self.num_layers, self.hidden_size)
        self.cell = mx.zeros(bidirection*self.num_layers, self.hidden_size)
        
        self._hh_proj = [Linear(hidden_size, 4*hidden_size, bias=bias) 
                         for _ in range(self.num_layers* bidirection)]

        

    def _extra_repr(self):
        return (
            f"input_dims={self._ih_proj.weight.shape[0]}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, \n\
                bidirectional={len(self._hh_proj)>self.num_layers}, bias={"bias" in self._in_proj}"
        )
    
    def _cell_fun(self, x, hidden):
        h, c = hidden
        xh = x + h
        ifo = mx.sigmoid(xh[...,-self.hidden_size:])
        g = mx.tanh(xh[...,:-self.hidden_size])
        i, f, o = mx.split(ifo, 3, axis=-1)
        c = f*c + i*g
        h = o*mx.tanh(c)
        return (h, c)

    def __call__(self, x, hidden=None, cell=None):
        x = self._ih_proj(x)

        all_hidden, all_cell= [], []
        last_hidden = None
        batch_size, seqlen = x.shape[:2]

        self.hidden = hidden or self.hidden
        self.cell = cell or self.hidden
  
        for hh_idx, hh_proj in enumerate(self._hh_proj):
            ih = self.hidden[hh_idx, :], self.cell[hh_idx,:]
            steps_hidden, steps_cell = [], []
            for idx in range(seqlen):
                if hh_idx ==0:
                    ix = x[..., idx, :] 
                elif hh_idx == self.num_layers:
                    ix = x[..., seqlen-1 - idx, :]
                else:
                    ix = last_hidden[idx]

                ih = self._cell_fun(hh_proj, ix, ih)
                steps_hidden.append(ih[0])
                steps_cell.append(ih[1])
            
            if hh_idx == self.num_layers - 1:
                all_hidden.extend(steps_hidden)
                all_cell.extend(steps_cell)
            elif hh_idx == 2*self.num_layers - 1:
                all_hidden.extend(steps_hidden[::-1])
                all_cell.extend(steps_cell[::-1])
            
            last_hidden = steps_hidden

        return (mx.stack(all_hidden).reshape((seqlen, batch_size, -1)), 
                mx.stack(all_cell).reshape((seqlen, batch_size, -1)))
