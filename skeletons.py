import pyopencl as cl
import numpy as np

# (Implementación de Esqueletos), Contiene los esqueletos de Pipeline y Master-Slave.
class Pipeline:
    """Esqueleto Pipeline: Ejecuta una serie de tareas en secuencia"""
    def __init__(self, env):
        self.env = env
        self.tasks = []

    def add_task(self, kernel_name, global_work_size):
        kernel = getattr(self.env.program, kernel_name)
        self.tasks.append((kernel, global_work_size))

    def execute(self, input_data, width, height):
        # Buffer inicial
        in_buf = self.env.create_buffer(input_data, cl.mem_flags.READ_ONLY)
        # Buffer de salida (asumimos grises o igual tamaño para simplificar el flujo)
        out_buf = cl.Buffer(self.env.context, cl.mem_flags.READ_WRITE, input_data.nbytes)

        for kernel, gws in self.tasks:
            kernel(self.env.queue, gws, None, in_buf, out_buf, np.int32(width), np.int32(height))
            # El output de uno es el input del siguiente
            in_buf = out_buf 
        
        result = np.empty_like(input_data)
        if len(result.shape) > 2: # Si es color y pasó a gris, ajustar shape
            result = np.empty((height, width), dtype=np.uint8)
            
        cl.enqueue_copy(self.env.queue, result, out_buf)
        return result

class MasterSlave:
    """Esqueleto Master-Slave: Distribuye datos para procesamiento paralelo masivo"""
    def __init__(self, env):
        self.env = env

    def run(self, kernel_name, data_chunks, width, height):
        # El Master coordina el envío de múltiples bloques a la GPU/FPGA
        results = []
        kernel = getattr(self.env.program, kernel_name)
        
        for chunk in data_chunks:
            buf_in = self.env.create_buffer(chunk)
            buf_out = cl.Buffer(self.env.context, cl.mem_flags.READ_WRITE, chunk.nbytes)
            kernel(self.env.queue, chunk.shape, None, buf_in, buf_out, np.int32(width), np.int32(height))
            res = np.empty_like(chunk)
            cl.enqueue_copy(self.env.queue, res, buf_out)
            results.append(res)
        return results
