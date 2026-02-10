import numpy as np
from PIL import Image
from askelhcore.core import SkeletonEnv
from askelhcore.skeletons import Pipeline

# (Programa de Ejemplo), Simula el papi-skeletoncore-main-program_ver_1-4.cpp.
def main():
    # 1. Inicializar ambiente
    env = SkeletonEnv()

    # 2. Cargar imagen de prueba (simulando entrada)
    # Generamos una imagen dummy de 512x512 RGB si no hay una real
    width, height = 512, 512
    input_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # 3. Definir Esqueleto Pipeline
    # Tarea 1: Grayscale -> Tarea 2: Sobel Edge Detection
    pipe = Pipeline(env)
    pipe.add_task("grayscale_kernel", (width, height))
    pipe.add_task("sobel_kernel", (width, height))

    # 4. Ejecutar
    print("Ejecutando Pipeline: Grayscale + Sobel...")
    result = pipe.execute(input_image, width, height)

    # 5. Guardar resultado
    res_img = Image.fromarray(result)
    res_img.save("output_sobel.jpg")
    print("Proceso completado. Imagen guardada como output_sobel.jpg")

if __name__ == "__main__":
    main()
