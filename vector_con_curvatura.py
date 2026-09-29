"""Módulo para el cálculo de features de curvatura por dedo y construcción del vector ampliado de 136 dimensiones.

Fórmula exacta de curvatura:
Para cada uno de los 5 dedos (Pulgar, Índice, Medio, Anular, Meñique), se seleccionan 3 landmarks en coordenadas
world métricas (landmarks_world, invariantes a escala de cámara y perspectiva):
  - Pulgar:  Base = 2 (MCP), Articulación media = 3 (IP),  Punta = 4 (TIP)
  - Índice:  Base = 5 (MCP), Articulación media = 6 (PIP), Punta = 8 (TIP)
  - Medio:   Base = 9 (MCP), Articulación media = 10 (PIP), Punta = 12 (TIP)
  - Anular:  Base = 13 (MCP), Articulación media = 14 (PIP), Punta = 16 (TIP)
  - Meñique: Base = 17 (MCP), Articulación media = 18 (PIP), Punta = 20 (TIP)

A partir del punto de la articulación media B, se definen los vectores de los segmentos adyacentes:
  v1 = P_base - P_media   (vector hacia la base)
  v2 = P_punta - P_media  (vector hacia la punta)

El ángulo interno theta en la articulación se calcula mediante el producto punto normalizado:
  cos(theta) = clip( (v1 · v2) / (||v1|| * ||v2||), -1.0, 1.0 )
  theta = arccos(cos(theta))   [en radianes, rango [0, pi]]

Interpretación biomecánica:
  - Dedo extendido (recto):     theta ≈ pi ≈ 3.1416 rad (180°)
  - Dedo en gancho / curvado:   theta ≈ pi/2 ≈ 1.5708 rad (90°)
  - Dedo totalmente flexionado: theta < 1.3 rad (< 75°)

Estructura del vector de 136 valores (126 + 10):
  - Índices 0..62:    Mano izquierda (63 valores normalizados con normalize_keypoints).
  - Índices 63..125:  Mano derecha (63 valores normalizados con normalize_keypoints).
  - Índices 126..130: 5 valores de curvatura de la mano izquierda (Pulgar, Índice, Medio, Anular, Meñique).
  - Índices 131..135: 5 valores de curvatura de la mano derecha (Pulgar, Índice, Medio, Anular, Meñique).
  (Si una mano está ausente, sus 63 valores de posición y sus 5 valores de curvatura son 0.0).
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from sign_classifier import normalize_keypoints, hand_to_feature_vector

# Tríos de landmarks (Base, Media, Punta) en MediaPipe Hands
DEDOS_TRIPLETS: Tuple[Tuple[int, int, int], ...] = (
    (2, 3, 4),    # 0: Pulgar (MCP -> IP -> TIP)
    (5, 6, 8),    # 1: Índice (MCP -> PIP -> TIP)
    (9, 10, 12),  # 2: Medio (MCP -> PIP -> TIP)
    (13, 14, 16), # 3: Anular (MCP -> PIP -> TIP)
    (17, 18, 20), # 4: Meñique (MCP -> PIP -> TIP)
)

N_FEATURES_126 = 126
N_FEATURES_136 = 136
N_FINGERS = 5
N_HAND_FEATURES = 63


def calcular_curvatura_mano(
    landmarks_world_hand: np.ndarray,
    normalizar_01: bool = False,
) -> np.ndarray:
    """Calcula el ángulo interno de curvatura para los 5 dedos a partir de los landmarks world (21, 3).

    Args:
        landmarks_world_hand: Array de shape (21, 3) o (21, >=3) con coordenadas métricas world.
        normalizar_01: Si es True, divide el ángulo en radianes entre pi para que esté en [0, 1].
                       Por defecto es False (radianes directos [0, pi]).

    Returns:
        Array de 5 valores float32 con los ángulos de los dedos [Pulgar, Índice, Medio, Anular, Meñique].
    """
    curvaturas = np.zeros(N_FINGERS, dtype=np.float32)

    if landmarks_world_hand is None or len(landmarks_world_hand) < 21:
        return curvaturas

    for idx, (idx_base, idx_media, idx_punta) in enumerate(DEDOS_TRIPLETS):
        p_base = landmarks_world_hand[idx_base, :3]
        p_media = landmarks_world_hand[idx_media, :3]
        p_punta = landmarks_world_hand[idx_punta, :3]

        v1 = p_base - p_media
        v2 = p_punta - p_media

        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_th = float(np.dot(v1, v2) / (norm1 * norm2))
            cos_th = float(np.clip(cos_th, -1.0, 1.0))
            ang = float(np.arccos(cos_th))
            curvaturas[idx] = (ang / np.pi) if normalizar_01 else ang
        else:
            curvaturas[idx] = 0.0

    return curvaturas


def construir_vector_136_frame(
    landmarks_image_left: Optional[np.ndarray],
    landmarks_world_left: Optional[np.ndarray],
    landmarks_image_right: Optional[np.ndarray],
    landmarks_world_right: Optional[np.ndarray],
    normalizar_curvatura: bool = False,
) -> np.ndarray:
    """Construye un vector de 136 dimensiones para un frame a partir de los landmarks de ambas manos.

    Args:
        landmarks_image_left: (21, >=2) o None si mano izquierda ausente.
        landmarks_world_left: (21, 3) o None.
        landmarks_image_right: (21, >=2) o None si mano derecha ausente.
        landmarks_world_right: (21, 3) o None.
        normalizar_curvatura: Si divide los ángulos entre pi (rango [0, 1]).

    Returns:
        Vector float32 de 136 elementos:
          [0..62]:    Mano izquierda (63 valores normalizados)
          [63..125]:  Mano derecha (63 valores normalizados)
          [126..130]: Curvatura mano izquierda (5 valores)
          [131..135]: Curvatura mano derecha (5 valores)
    """
    out = np.zeros(N_FEATURES_136, dtype=np.float32)

    # Mano izquierda (Slot 0)
    if landmarks_image_left is not None and landmarks_world_left is not None:
        raw_left = hand_to_feature_vector(landmarks_image_left[:, :2], landmarks_world_left)
        norm_left = normalize_keypoints(raw_left)
        curv_left = calcular_curvatura_mano(landmarks_world_left, normalizar_01=normalizar_curvatura)
        out[0:63] = norm_left
        out[126:131] = curv_left

    # Mano derecha (Slot 1)
    if landmarks_image_right is not None and landmarks_world_right is not None:
        raw_right = hand_to_feature_vector(landmarks_image_right[:, :2], landmarks_world_right)
        norm_right = normalize_keypoints(raw_right)
        curv_right = calcular_curvatura_mano(landmarks_world_right, normalizar_01=normalizar_curvatura)
        out[63:126] = norm_right
        out[131:136] = curv_right

    return out


def frame_crudo_a_vector_136(
    hand_labels_frame: np.ndarray,          # (2,)
    landmarks_image_frame: np.ndarray,      # (2, 21, >=2)
    landmarks_world_frame: np.ndarray,      # (2, 21, 3)
    normalizar_curvatura: bool = False,
) -> np.ndarray:
    """Extrae el vector de 136 valores directamente del formato crudo de un frame guardado en los .npz."""
    lm_img_l = landmarks_image_frame[0] if hand_labels_frame[0] != "" else None
    lm_wrd_l = landmarks_world_frame[0] if hand_labels_frame[0] != "" else None
    lm_img_r = landmarks_image_frame[1] if hand_labels_frame[1] != "" else None
    lm_wrd_r = landmarks_world_frame[1] if hand_labels_frame[1] != "" else None

    return construir_vector_136_frame(
        lm_img_l, lm_wrd_l, lm_img_r, lm_wrd_r, normalizar_curvatura=normalizar_curvatura
    )
