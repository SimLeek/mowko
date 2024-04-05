"""Module to help with backprop, optim, etc."""
import abc

from mowko.gpu import GPUManager, get_shader
from typing import List
import kp


class Module(object):
    """Module to help with backprop, optim, etc."""

    def __init__(self, gpu: GPUManager):
        """Setup module requirements."""
        self.gpu = gpu

        self.has_forward = False
        self.has_backward = False
        self.has_optim = False

        # for large constant arrays
        self.optim_setup_buffers: List[kp.Tensor] = []
        self.forward_setup_buffers: List[kp.Tensor] = []
        self.backward_setup_buffers: List[kp.Tensor] = []

        # for input to the GPU.
        self.optim_input_buffers: List[kp.Tensor] = []
        self.forward_input_buffers: List[kp.Tensor] = []
        self.backward_input_buffers: List[kp.Tensor] = []

        # for output from the GPU
        #   You can take output from any buffer for debugging, but these are the outputs during normal operation.
        self.optim_output_buffers: List[kp.Tensor] = []
        self.forward_output_buffers: List[kp.Tensor] = []
        self.backward_output_buffers: List[kp.Tensor] = []

    def setup_check(self):
        """Validate module in general."""
        assert any(
            [self.has_forward, self.has_backward, self.has_optim]
        ), "Module must do *something*."
        assert any(
            [
                len(self.optim_input_buffers) != 0,
                len(self.forward_input_buffers) != 0,
                len(self.backward_input_buffers) != 0,
            ]
        ), "Module must take input."
        assert any(
            [
                len(self.optim_output_buffers) != 0,
                len(self.forward_output_buffers) != 0,
                len(self.backward_output_buffers) != 0,
            ]
        ), "Module must give output."

    @abc.abstractmethod
    def forward_ops(self):
        """Record all forward kompute ops here."""
        raise NotImplementedError()

    @abc.abstractmethod
    def backward_ops(self):
        """Record all backwards kompute ops here."""
        raise NotImplementedError()

    @abc.abstractmethod
    def optim_ops(self):
        """Record all optim kompute ops here."""
        raise NotImplementedError()
