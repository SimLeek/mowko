"""Serialization util."""
from io import BytesIO

import numpy as np


def serialize_buffer(buf):
    """Serialize a numpy array into a BytesIO object."""
    data = buf.data()
    bytes_io = BytesIO()
    np.savez_compressed(bytes_io, data=data)
    bytes_io.seek(0)
    return bytes_io


def deserialize_buffer(buf) -> np.ndarray:
    """Deserialize a BytesIO object into a numpy array."""
    buffer = np.load(buf, allow_pickle=False)["data"]
    return buffer
