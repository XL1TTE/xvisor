from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from platform import processor
from torch import device, cuda 


@dataclass(frozen=True)
class GpuInfo:
    model: str
    vram: float


@dataclass(frozen=True)
class CpuInfo:
    model: str


@dataclass(frozen=True)
class Device(ABC):
    device: device

    @staticmethod
    def create_gpu() -> GpuDevice:
        t_device = device("cuda:0")
        gpu_idx = t_device.index or 0

        t_device_info = GpuInfo(
            model=cuda.get_device_name(gpu_idx),
            vram=round(cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3), 2),
        )
        return GpuDevice(device=t_device, info=t_device_info)

    @staticmethod
    def create_cpu() -> CpuDevice:
        t_device = device("cpu")
        t_device_info = CpuInfo(model=processor() or "Unknown CPU")
        return CpuDevice(device=t_device, info=t_device_info)


@dataclass(frozen=True)
class CpuDevice(Device):
    info: CpuInfo


@dataclass(frozen=True)
class GpuDevice(Device):
    info: GpuInfo


class DeviceProvider:
    __device: CpuDevice | GpuDevice

    def __init__(self) -> None:
        self.__device = Device.create_gpu() if cuda.is_available() else Device.create_cpu()

    def get_device(self) -> device:
        return self.__device.device

    def get_device_info(self) -> GpuInfo | CpuInfo:
        return self.__device.info
