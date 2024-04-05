"""Query all GPU capabilities."""

import vulkan as vk
import pprint
from typing import Iterable
import weakref as _weakref
from vulkan._vulkancache import ffi

_weakkey_dict = _weakref.WeakKeyDictionary()  # type: ignore


def _cast_ptr2(x, _type):
    if isinstance(x, ffi.CData):
        if _type.item == ffi.typeof(x) or (
            _type.item.cname == "void" and ffi.typeof(x).kind in ["struct", "union"]
        ):
            return ffi.addressof(x), x
        return x, x

    if isinstance(x, Iterable):
        if _type.item.kind == "pointer":
            ptrs = [_cast_ptr(i, _type.item) for i in x]
            ret = ffi.new(_type.item.cname + "[]", [i for i, _ in ptrs])
            _weakkey_dict[ret] = tuple(i for _, i in ptrs if i != ffi.NULL)
        else:
            ret = ffi.new(_type.item.cname + "[]", x)

        return ret, ret

    return ffi.cast(_type, x), x


def _cast_ptr3(x, _type):
    if isinstance(x, str):
        try:
            x = x.encode("ascii")
        except UnicodeEncodeError:
            x = x.encode("utf-8")
    return _cast_ptr2(x, _type)


_cast_ptr = _cast_ptr3


def _wrap_vkGetPhysicalDeviceProperties2(fn):
    def vkGetPhysicalDeviceProperties2(
        physicalDevice,
        pProperties=None,
    ):

        custom_return = True
        if not pProperties:
            pProperties = ffi.new("VkPhysicalDeviceProperties2*")
            custom_return = False

        result = _callApi(fn, physicalDevice, pProperties)

        if custom_return:
            return pProperties

        return pProperties[0]

    return vkGetPhysicalDeviceProperties2


def _auto_handle(x, _type):
    if x is None:
        return ffi.NULL
    if _type.kind == "pointer":
        ptr, _ = _cast_ptr(x, _type)
        return ptr
    return x


def _callApi(fn, *args):
    fn_args = [_auto_handle(i, j) for i, j in zip(args, ffi.typeof(fn).args)]
    return fn(*fn_args)


def dumpdict(obj, dict_part=None):
    """Dump a c object as a dictionary."""
    empty = True
    if dict_part is None:
        dict_part = dict()
    for a in dir(obj):
        empty = False
        val = getattr(obj, a)
        if isinstance(val, (int, float, str, list, dict, set)):
            dict_part[str(a)] = val
        else:
            dict_part[str(a)] = dict()
            dict_part[str(a)] = dumpdict(val, dict_part[str(a)])
    if empty:
        if "void *" in str(obj):
            dict_part = str(obj)
        elif "char[" in str(obj):
            c_str = b"".join([c for c in obj])
            dict_part = c_str
        elif "[" in str(obj):
            dict_part = [f for f in obj]
        else:
            dict_part = None
    return dict_part


class InstanceProcAddr(object):
    """Wrap InstanceProcAddr."""

    T = None

    def __init__(self, func):
        """Wrap func."""
        self.__func = func

    def __call__(self, *args, **kwargs):
        """Wrap all vulkan functions."""
        funcName = self.__func.__name__
        func = InstanceProcAddr.procfunc(funcName)
        if func:
            return func(*args, **kwargs)
        else:
            return vk.VK_ERROR_EXTENSION_NOT_PRESENT

    @staticmethod
    def procfunc(funcName):
        """Call Vulkan procfunc."""
        fn = _callApi(vk.lib.vkGetInstanceProcAddr, InstanceProcAddr.T, funcName)
        # if fn == ffi.NULL:
        #    raise ProcedureNotFoundError()
        # if not pName in _instance_ext_funcs:
        #    raise ExtensionNotSupportedError()
        fn = ffi.cast("PFN_" + "vkGetPhysicalDeviceProperties2", fn)
        return _wrap_vkGetPhysicalDeviceProperties2(fn)


@InstanceProcAddr
def vkGetPhysicalDeviceProperties2(physicalDevice, pProperties=None):
    """Wrapper for vkGetPhysicalDeviceProperties2."""
    pass


def get_gpu_info(gpu_number=None):
    """Get available vulkan info for any or all GPUs."""
    appInfo = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="Hello Triangle",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0,
    )

    layers = vk.vkEnumerateInstanceLayerProperties()
    layers = [l.layerName for l in layers]

    if "VK_LAYER_KHRONOS_validation" in layers:
        layers = ["VK_LAYER_KHRONOS_validation"]
    elif "VK_LAYER_LUNARG_standard_validation" in layers:
        layers = ["VK_LAYER_LUNARG_standard_validation"]
    else:
        layers = []

    extensions = vk.vkEnumerateInstanceExtensionProperties(None)
    extensions = [e.extensionName for e in extensions]

    extensions = ["VK_KHR_surface", "VK_EXT_debug_report"]

    createInfo = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=0,
        pApplicationInfo=appInfo,
        enabledLayerCount=len(layers),
        ppEnabledLayerNames=layers,
        enabledExtensionCount=len(extensions),
        ppEnabledExtensionNames=extensions,
    )
    instance = vk.vkCreateInstance(createInfo, None)
    InstanceProcAddr.T = instance
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)

    gpu_info = {}
    for i, physical_device in enumerate(physical_devices):
        device_properties = vk.vkGetPhysicalDeviceProperties(physical_device)
        device_name = device_properties.deviceName

        if gpu_number is None or gpu_number in [device_properties.deviceID, i]:
            physical_devices_extensions = vk.vkEnumerateDeviceExtensionProperties(
                physical_device, None
            )

            extensions_info = [
                {"name": e.extensionName, "version": e.specVersion}
                for e in physical_devices_extensions
            ]

            subgroup_props = vk.VkPhysicalDeviceSubgroupProperties(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                pNext=None,
                subgroupSize=None,
                supportedStages=None,
                supportedOperations=None,
                quadOperationsInAllStages=None,
            )

            devprops2 = vk.VkPhysicalDeviceProperties2(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                pNext=subgroup_props,
                properties=None,
            )

            props = vkGetPhysicalDeviceProperties2(physical_device, devprops2)

            sub_props = ffi.cast("VkPhysicalDeviceSubgroupProperties*", props.pNext)

            gpu_info[i] = {
                "device_name": device_name,
                "device_id": device_properties.deviceID,
                "extensions": extensions_info,
                "limits": dumpdict(props.properties.limits),
                "subgroup_properties": {
                    "subgroup_size": sub_props.subgroupSize,
                    "supported_stages": sub_props.supportedStages,
                    "supported_operations": sub_props.supportedOperations,
                    "quad_operations_in_all_stages": sub_props.quadOperationsInAllStages,
                },
            }

    vk.vkDestroyInstance(instance, None)
    return gpu_info


if __name__ == "__main__":
    gpu_info = get_gpu_info(0)
    for gpu_id, info in gpu_info.items():
        print("GPU:")
        pprint.pp({gpu_id})
        print("\tDevice Name:")
        pprint.pp(info["device_name"])
        print("\tDevice ID:")
        pprint.pp(info["device_id"])
        print("\tExtensions:")
        pprint.pp(info["extensions"])
        print("\tLimits:")
        pprint.pp(info["limits"])
        print("\tSubgroup Properties:")
        pprint.pp(info["subgroup_properties"])
        pprint.pp("---------------------------------------------")
