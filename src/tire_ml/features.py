"""Medidas por cuadro a partir de los puntos faciales de MediaPipe Face Landmarker.

Todas las funciones reciben los puntos en PÍXELES (x, y), no normalizados:
MediaPipe entrega x y y entre 0 y 1, pero escalados de forma distinta por el
ancho y el alto de la imagen, y eso deforma el EAR si no se convierten.
"""
from __future__ import annotations

import math

import numpy as np

# Índices de la malla de 478 puntos de MediaPipe, en orden p1..p6:
# p1 y p4 son las esquinas del ojo; (p2, p6) y (p3, p5) son pares vertical arriba/abajo.
OJO_IZQ = (33, 160, 158, 133, 153, 144)
OJO_DER = (362, 385, 387, 263, 373, 380)

# Boca: labio interior superior, labio interior inferior, comisura izq., comisura der.
BOCA = (13, 14, 78, 308)


def a_pixeles(landmarks, ancho: int, alto: int) -> np.ndarray:
    """Convierte la lista de NormalizedLandmark de MediaPipe a un arreglo (N, 2) en píxeles."""
    return np.array([(p.x * ancho, p.y * alto) for p in landmarks], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def ear_ojo(puntos: np.ndarray, indices: tuple[int, ...]) -> float:
    """Eye Aspect Ratio de un ojo: (|p2-p6| + |p3-p5|) / (2·|p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = (puntos[i] for i in indices)
    ancho = _dist(p1, p4)
    if ancho == 0:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * ancho)


def ear(puntos: np.ndarray) -> float:
    """EAR promedio de ambos ojos."""
    return (ear_ojo(puntos, OJO_IZQ) + ear_ojo(puntos, OJO_DER)) / 2.0


def mar(puntos: np.ndarray) -> float:
    """Mouth Aspect Ratio: apertura vertical / ancho de la boca. Sube con los bostezos."""
    arriba, abajo, izq, der = (puntos[i] for i in BOCA)
    ancho = _dist(izq, der)
    if ancho == 0:
        return 0.0
    return _dist(arriba, abajo) / ancho


def postura_cabeza(matriz) -> tuple[float, float, float]:
    """(pitch, yaw, roll) en grados a partir de la matriz 4x4 de transformación facial.

    pitch: cabeceo arriba/abajo · yaw: giro a los lados · roll: ladeo.
    Los signos dependen de la convención de MediaPipe: verifícalos con la webcam
    (baja la cabeza y mira qué signo toma el pitch) antes de fijar umbrales.
    """
    r = np.asarray(matriz, dtype=np.float64)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll
