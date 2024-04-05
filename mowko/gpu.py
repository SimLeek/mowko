"""GPU manager and shaders."""

import os
import subprocess
import sys
from mowko.vk_extensions import get_gpu_info
import kp


class GPUManager(kp.Manager):
    """Manages a GPU device."""

    def __init__(
        self,
        device: int = 0,
        family_queue_indices=None,
        desired_extensions=None,
        *args,
        **kwargs
    ):
        """Initialize Kompute GPU manager with some extra info."""
        if family_queue_indices is not None and desired_extensions is not None:
            super().__init__(device, family_queue_indices, desired_extensions)
        else:
            super().__init__(device)

        self.max_workgroup_invocations = self.manager.get_device_properties()[
            "max_work_group_invocations"
        ]
        # todo: finish this PR to get it from kompute: https://github.com/KomputeProject/kompute/issues/360
        # this variable is specifically necessary for reductions ops often used in sparse shaders:
        self.max_compute_shared_memory_size = get_gpu_info(device)[device]["limits"][
            "maxComputeSharedMemorySize"
        ]  # typical: 49152
        # Useful for defining shared array sizes:
        self.max_push_constant_size = get_gpu_info(device)[device]["limits"][
            "maxPushConstantsSize"
        ]  # typical: 256
        # Note^: AMD is often 128 while NVidia is 256, so some shaders may need to be split on AMD.


def get_shader(filename):
    """Compile a compute shader if needed, otherwise read the spir-v code."""
    if filename.endswith(".glsl") or filename.endswith(".comp"):
        spv_filename = filename[:-5] + ".spv"
        if not os.path.exists(spv_filename):
            try:
                subprocess.run(
                    ["glslc", filename, "-o", spv_filename],
                    check=True,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print("glslc command failed with output:")
                print(e.stdout)
                print(e.stderr, file=sys.stderr)
                raise e

        with open(spv_filename, "rb") as f:
            shader = f.read()
    else:
        raise ValueError(
            "Invalid file extension. Filename must end with .glsl or .comp"
        )

    return shader
