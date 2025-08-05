# Detección de Cáncer Cerebral con Deep Learning (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-0.15.2-orange?logo=pytorch)](https://pytorch.org/vision/stable/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-yellow?logo=matplotlib)](https://matplotlib.org/)

## Descripción

Este proyecto implementa una red neuronal convolucional (CNN) para la detección y clasificación de cáncer cerebral a partir de imágenes de resonancia magnética (MRI), utilizando PyTorch. El objetivo es asistir en el diagnóstico médico mediante el reconocimiento automático de distintos tipos de tumores cerebrales.

## Tecnologías utilizadas

- **Python 3.12**
- **PyTorch** (entrenamiento del modelo y arquitectura CNN)
- **Torchvision** (transformaciones de imágenes y datasets)
- **Matplotlib** (visualización de resultados)

## Estructura del dataset

El conjunto de datos está organizado en tres carpetas principales, ya preprocesadas:

- `data/train` (70%): imágenes para entrenamiento
- `data/val` (15%): imágenes para validación
- `data/test` (15%): imágenes para test

## Clases del modelo

El modelo es un clasificador multiclase, capaz de identificar:

- **Clase 0:** Tumor tipo Glioma
- **Clase 1:** Tumor tipo Meningioma
- **Clase 2:** No tumor (imagen sana)
- **Clase 3:** Tumor tipo Pituitary

## Pipeline general

1. **Comprobación de GPU:** Se verifica la disponibilidad de CUDA.
2. **Transformaciones:** Se aplican aumentos de datos y normalización a las imágenes.
3. **Carga de datos:** Uso de `torchvision.datasets.ImageFolder` y `DataLoader`.
4. **Definición de la arquitectura:** CNN personalizada con varias capas convolucionales y fully-connected.
5. **Entrenamiento:** Optimización con Adam y scheduler de tasa de aprendizaje.
6. **Validación y test:** Se evalúa el accuracy en el conjunto de validación y test.
7. **Guardado y carga del modelo:** Se almacena el modelo entrenado para su posterior uso.

## Ejemplo de entrenamiento

```python
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
# Entrenamiento del modelo...
```

## Resultados

- **Precisión en test:** ~91%
- El modelo interpreta y predice una de las 4 clases para cada imagen MRI.

## Uso

1. Clona el repositorio y coloca las imágenes en las carpetas correspondientes.
2. Ejecuta el notebook `cnn_cancer_brain.ipynb` para entrenar y evaluar el modelo.
3. Puedes cargar pesos previamente entrenados y hacer predicciones sobre nuevas imágenes MRI.

## Créditos

Proyecto desarrollado para fines educativos y de investigación en el área de diagnóstico médico asistido por IA.

---
