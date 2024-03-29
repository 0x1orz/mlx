# Copyright © 2023-2024 Apple Inc.

from typing import Callable, Optional, override
from itertools import chain
import mlx.core as mx
import numpy as np
from mlx.nn.layers.activations import tanh
from mlx.nn.layers import Linear
from mlx.nn.layers.base import Module

class RecurrentBase(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        bias: bool = True,):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_bidirections = 2 if bidirectional else 1
        self.num_layers = num_layers

        self._ih_proj: Module = None

        assert num_layers >= 1, "num_layers of the hidden layers should be more than 1."
        self.h_0 : mx.array = mx.zeros((self.num_bidirections * self.num_layers, hidden_size))
        self.h_n : mx.array = None
        self._hh_proj : Module = None

    @property
    def weight_ih(self):
        return [layer.weight for layer in self._ih_proj]
    @property
    def weight_hh(self):
        weights = [layer.weight for layer in self._hh_proj[:self.num_layers]]
        return weights
    @property
    def weight_hh_reverse(self):
        weights = [layer.weight for layer in self._hh_proj[self.num_layers:]]
        return weights
    
    def _extra_repr(self):
        return (
            f"input_dims={[weight.shape for weight in self.weight_ih]},\n"
            f"hidden_size={[weight.shape for weight in self.weight_hh]},\n\
                num_layers={self.num_layers}, bias={"bias" in self._in_proj},\n\
                bidirectional={len(self._hh_proj)>self.num_layers}"
        )
    
    @override
    def _cell_fun(self, hh_proj, x, states):
        raise NotImplementedError
    
class RNN(RecurrentBase):
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
        super().__init__(input_size, hidden_size, num_layers, bidirectional, bias)

        _ih_input_size = np.array([[input_size] + [hidden_size]* (num_layers - 1)] * self.num_bidirections).flatten()
        self._ih_proj = [Linear(m, hidden_size, bias=bias) for m in _ih_input_size]
        _hh_input_size = [hidden_size] * self.num_bidirections * self.num_layers
        self._hh_proj = [Linear(m, hidden_size, bias) for m in _hh_input_size]

        self.nonlinearity = nonlinearity or tanh
        if not callable(self.nonlinearity):
            raise ValueError(
                f"Nonlinearity must be callable. Current value: {nonlinearity}."
            )

    def _cell_fun(self, hh_proj, x, states):
        h = hh_proj(states) + x
        h = self.nonlinearity(h)
        return h

    def __call__(self, x, hidden=None):
        self.h_0 = hidden or self.h_0
        all_hidden, h_n = [], []
        last_h = None
        batch_size, seq_len = x.shape[:2]
        for proj_idx, hh_proj in enumerate(self._hh_proj):
            steps_h = []
            ih = mx.expand_dims(self.h_0[proj_idx, :], 0)
            for idx in range(seq_len):

                if proj_idx == 0:
                    ix = x[..., idx, :] 
                elif proj_idx == self.num_layers: 
                    ix = x[..., seq_len-1 - idx, :]
                else:
                    ix = last_h[idx]
                ix = self._ih_proj[proj_idx](ix)
                ih = self._cell_fun(hh_proj, ix, ih)
                steps_h.append(ih)
            h_n.append(ih)

            if proj_idx == self.num_layers - 1:
                all_hidden.extend(steps_h)
            elif proj_idx == 2*self.num_layers - 1:
                all_hidden.extend(steps_h[::-1])

            last_h = steps_h
        self.h_n = mx.stack(h_n)

        return mx.stack(all_hidden).reshape((seq_len, batch_size, -1)), self.h_n

class GRU(RecurrentBase):
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
        super().__init__(input_size, hidden_size, num_layers, bidirectional, bias)

        repeat_num = 3
        _ih_input_size = np.array([[input_size] + [hidden_size]* (num_layers - 1)] * self.num_bidirections).flatten()
        self._ih_proj = [Linear(m, repeat_num * hidden_size, bias=bias) for m in _ih_input_size]
        _hh_input_size = [hidden_size] * self.num_bidirections * self.num_layers
        self._hh_proj = [Linear(m, repeat_num * hidden_size, bias=bias) for m in _hh_input_size]

    
    def _cell_fun(self, hh_proj, x, states):
        x_rz, x_n = x[..., :-self.hidden_size], x[..., -self.hidden_size:]
        h = hh_proj(states)
        h_rz, h_n = h[..., :-self.hidden_size], h[..., -self.hidden_size:]
        rz = mx.sigmoid(x_rz + h_rz)
        r, z= mx.split(rz, 2, axis=-1)
        n = mx.tanh(x_n + r*h_n)
        h = (1-z)*n + z*h_n
        return h
        
    def __call__(self, x, hidden=None):
        self.h_0 = hidden or self.h_0
        all_hidden, h_n = [], []
        last_h = None
        batch_size, seq_len = x.shape[:2]
        for proj_idx, hh_proj in enumerate(self._hh_proj):
            ih = mx.expand_dims(self.h_0[proj_idx, :], 0)
            step_h = []
            for idx in range(seq_len):
                if proj_idx ==0:
                    ix = x[..., idx, :] 
                elif proj_idx == self.num_layers:
                    ix = x[..., seq_len-1 - idx, :]
                else:
                    ix =last_h[idx]
                ix = self._ih_proj[proj_idx](ix)
                ih = self._cell_fun(hh_proj, ix, ih)
                step_h.append(ih)
            h_n.append(ih)
             
            if proj_idx == self.num_layers - 1:
                all_hidden.extend(step_h)
            elif proj_idx == 2*self.num_layers - 1:
                all_hidden.extend(step_h[::-1])

            last_h = step_h

        self.h_n = mx.stack(h_n)

        return mx.stack(all_hidden).reshape((seq_len, batch_size, -1)), self.h_n
    

class LSTM(RecurrentBase):
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
        super().__init__(input_size, hidden_size, num_layers, bidirectional, bias)

        repeat_num = 4
        _ih_input_size = np.array([[input_size] + [hidden_size]* (num_layers - 1)] * self.num_bidirections).flatten()
        self._ih_proj = [Linear(m, repeat_num * hidden_size, bias=bias) for m in _ih_input_size]
        _hh_input_size = [hidden_size] * self.num_bidirections * self.num_layers
        self._hh_proj = [Linear(m, repeat_num * hidden_size, bias) for m in _hh_input_size]
        

        self.c_0 = mx.zeros_like(self.h_0)
        self.c_n = None
            
    def _cell_fun(self,hh_proj, x, states):
        h, c = states
        xh = hh_proj(h) + x
        ifo = mx.sigmoid(xh[..., :-self.hidden_size])
        g = mx.tanh(xh[..., -self.hidden_size:])
        i, f, o = mx.split(ifo, 3, axis=-1)
        c = f*c + i*g
        h = o*mx.tanh(c)
        return (h, c)

    def __call__(self, x, hidden=None, cell=None):
    
        all_hidden, all_cell = [], []
        h_n, c_n = [], []
        last_h = None
        batch_size, seq_len = x.shape[:2]

        h_0 = hidden or self.h_0
        c_0 = cell or self.h_0
  
        for proj_idx, hh_proj in enumerate(self._hh_proj):
            ih = (mx.expand_dims(h_0[proj_idx, :], 0), 
                  mx.expand_dims(c_0[proj_idx,:],0))
            steps_h, step_c = [], []
            for idx in range(seq_len):
                if proj_idx ==0:
                    ix = x[..., idx, :] 
                elif proj_idx == self.num_layers:
                    ix = x[..., seq_len-1 - idx, :]
                else:
                    ix = last_h[idx]

                ix = self._ih_proj[proj_idx](ix)
                ih = self._cell_fun(hh_proj, ix, ih)

                steps_h.append(ih[0])
                step_c.append(ih[1])

            h_n.append(ih[0])
            c_n.append(ih[1])

            if proj_idx == self.num_layers - 1:
                all_hidden.extend(steps_h)
                all_cell.extend(step_c)
            elif proj_idx == 2*self.num_layers - 1:
                all_hidden.extend(steps_h[::-1])
                all_cell.extend(step_c[::-1])
            
            last_h = steps_h
        self.h_n = mx.stack(h_n)
        self.c_n = mx.stack(c_n)

        return mx.stack(all_hidden).reshape((seq_len, batch_size, -1)), (self.h_n, self.c_n)