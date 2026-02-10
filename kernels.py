# kernels.py
OPENCL_KERNELS = """
// (Definición de Kernels), Aquí portamos las funciones de los archivos task_kernelX.hpp a código kernel de OpenCL.
// Kernel 1: Conversión a Grises (Inspirado en task_kernel1_ver_1-4.hpp)
__kernel void grayscale_kernel(__global const uchar* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int out_idx = y * width + x;
        uchar r = input[idx];
        uchar g = input[idx + 1];
        uchar b = input[idx + 2];
        output[out_idx] = (uchar)((r + g + b) / 3);
    }
}

// Kernel 2: Sobel Edge Detection (Inspirado en task_kernel2_ver_1-4.hpp)
__kernel void sobel_kernel(__global const uchar* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)] 
                   - 2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                   - input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];
                   
        float gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                   + input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
        
        output[y*width + x] = (uchar)clamp(sqrt(gx*gx + gy*gy), 0.0f, 255.0f);
    }
}

// Kernel 3: Rotación 90 Grados (Inspirado en task_kernel3_ver_1-4.hpp)
__kernel void rotate_90_kernel(__global const uchar* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        // Nueva posición: x' = height - 1 - y, y' = x
        output[x * height + (height - 1 - y)] = input[y * width + x];
    }
}
"""
