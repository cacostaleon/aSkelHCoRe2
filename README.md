# aSkelHCoRe: Framework de Esqueletos Algorítmicos Reconfigurables

**aSkelHCoRe** es un framework diseñado para el co-diseño en Sistemas Heterogéneos (RHCS), integrando FPGAs, CPUs y GPUs mediante el paradigma de esqueletos de Cole. Implementado en Python y OpenCL (vía PyOpenCL), ofrece abstracciones de alto nivel como los esqueletos **Pipeline** y **Master-Slave**.

## Requisitos
- Python 3.8+
- OpenCL 2.0+ Drivers y SDK del fabricante (Intel, AMD, NVIDIA o Xilinx).
- Librerías de Python: `pyopencl`, `numpy`, `Pillow` (para manejo de imágenes).

## Instalación
1. Clonar el repositorio.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
