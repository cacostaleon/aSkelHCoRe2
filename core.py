import pyopencl as cl
import numpy as np
from .kernels import OPENCL_KERNELS

# (Abstracción del Ambiente - Equivalente al Header), Define el objeto aSkelHCoRe para gestionar el contexto de OpenCL.
class SkeletonEnv:
    """Clase principal para gestionar el hardware heterogéneo (RHCS)"""
    def __init__(self, platform_idx=0, device_idx=0):
        # Selección de plataforma y dispositivo (CPU/GPU/FPGA)
        self.platforms = cl.get_platforms()
        self.platform = self.platforms[platform_idx]
        self.device = self.platform.get_devices()[device_idx]
        
        # Crear contexto compatible con OpenCL 2.0+
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Compilar Kernels
        self.program = cl.Program(self.context, OPENCL_KERNELS).build()
        print(f"[aSkelHCoRe] Usando Dispositivo: {self.device.name}")

    def create_buffer(self, data, flags=cl.mem_flags.READ_WRITE):
        mf = cl.mem_flags
        return cl.Buffer(self.context, flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
