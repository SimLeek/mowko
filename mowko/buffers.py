"""Wrappers for kompute tensors that support serialization."""

import numpy as np

from mowko.serial import deserialize_buffer, serialize_buffer
from mowko.gpu import GPUManager


class ImageBuffer(object):
    """Image Buffer."""

    def __init__(self, gpu: GPUManager, image):
        """Set up shape and buffer assuming a numpy image from OpenCV."""
        if isinstance(image, np.ndarray):
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.colors = image.shape[2]
            self.buffer = gpu.buffer(image)
        else:
            raise NotImplementedError(f"Unknown type: {type(image)}")

    def __setstate__(self, state):
        """Restore the state from serialized data."""
        self.height, self.width, self.colors = state[:3]
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        """Return state to be pickled."""
        return (self.height, self.width, self.colors, serialize_buffer(self.buffer))

    @property
    def size(self):
        """Get full data size."""
        return self.height * self.width * self.colors

    def set(self, image):
        """Set data to numpy array."""
        if isinstance(image, np.ndarray):
            self.buffer.data()[...] = image.flatten()
        else:
            raise NotImplementedError(f"Unknown type: {type(image)}")

    def get(self):
        """Return data as numpy array in the correct format."""
        return self.buffer.data().reshape(self.height, self.width, self.colors)


class GrayImageBuffer(object):
    """Grayscale Image Buffer."""

    def __init__(self, gpu: GPUManager, image):
        """Set up shape and buffer assuming a numpy image from OpenCV."""
        if isinstance(image, np.ndarray):
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.buffer = gpu.buffer(image)
        else:
            raise NotImplementedError(f"Unknown type: {type(image)}")

    def __setstate__(self, state):
        """Restore the state from serialized data."""
        self.height, self.width = state[:2]
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        """Return state to be pickled."""
        return (self.height, self.width, serialize_buffer(self.buffer))

    @property
    def size(self):
        """Get full data size."""
        return self.height * self.width

    def set(self, image):
        """Set data to numpy array."""
        if isinstance(image, np.ndarray):
            self.buffer.data()[...] = image.flatten()
        else:
            raise NotImplementedError(f"Unknown type: {type(image)}")

    def get(self):
        """Return data as numpy array in the correct format."""
        return self.buffer.data().reshape(self.height, self.width)
