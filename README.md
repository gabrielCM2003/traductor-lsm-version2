# Traductor de Lengua de Señas Mexicana (LSM)

Aplicación de escritorio que reconoce en tiempo real las letras del alfabeto de la
Lengua de Señas Mexicana a partir de la cámara, permitiendo construir palabras
gesto a gesto. Utiliza **MediaPipe** para detectar los puntos de la mano y un
modelo **ONNX** entrenado para clasificar cada letra.

## Características

- Detección de manos en tiempo real con MediaPipe Hand Landmarker.
- Clasificación del alfabeto LSM mediante un modelo ONNX.
- Interfaz gráfica construida con PyQt6.
- Suavizado temporal de predicciones y umbral de confianza configurable.
- Construcción de palabras letra por letra.

## Requisitos

- Python 3.10 o superior
- Una cámara web

## Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/TU_USUARIO/traductor-lsm.git
cd traductor-lsm

# 2. (Recomendado) Crea un entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instala las dependencias
pip install -r requirements.txt
```

## Uso

```bash
python señas.py
```

### Opciones disponibles

| Opción          | Descripción                                  |
|-----------------|----------------------------------------------|
| `--camera N`    | Índice de la cámara a usar (por defecto: 0)  |
| `--config RUTA` | Archivo de configuración JSON                |
| `--threshold X` | Confianza mínima de detección (0–1)          |
| `--max-hands N` | Número máximo de manos a detectar            |
| `-v`, `--verbose` | Muestra logs detallados                    |

Ejemplo:

```bash
python señas.py --camera 1 --threshold 0.7
```

## Estructura del proyecto

```
.
├── señas.py              # Aplicación principal (interfaz y lógica)
├── sign_classifier.py    # Carga del modelo ONNX y clasificación
├── lsm_alphabet.onnx     # Modelo entrenado
├── lsm_alphabet.onnx.data
├── lsm_labels.json       # Etiquetas de las letras
└── requirements.txt
```

## Autor

Josué Gabriel Cortés Muñoz
