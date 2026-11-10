import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tire_ml.features import OJO_DER, OJO_IZQ, a_pixeles, ear, ear_ojo, postura_cabeza  # noqa: E402


def ojo_sintetico(puntos, indices, cx, apertura, ancho=30.0):
    """Coloca un ojo de `ancho` px con una apertura vertical dada."""
    p1, p2, p3, p4, p5, p6 = indices
    puntos[p1] = (cx - ancho / 2, 100)
    puntos[p4] = (cx + ancho / 2, 100)
    puntos[p2] = (cx - 5, 100 - apertura / 2)
    puntos[p6] = (cx - 5, 100 + apertura / 2)
    puntos[p3] = (cx + 5, 100 - apertura / 2)
    puntos[p5] = (cx + 5, 100 + apertura / 2)


def cara(apertura):
    puntos = np.zeros((478, 2))
    ojo_sintetico(puntos, OJO_IZQ, 100, apertura)
    ojo_sintetico(puntos, OJO_DER, 200, apertura)
    return puntos


def test_ear_ojo_abierto_y_cerrado():
    assert ear(cara(9.0)) == pytest.approx(9.0 / 30.0)
    assert ear(cara(0.0)) == 0.0
    assert ear(cara(9.0)) > ear(cara(3.0))


def test_ear_ojo_degenerado_no_divide_por_cero():
    assert ear_ojo(np.zeros((478, 2)), OJO_IZQ) == 0.0


def test_a_pixeles_escala_x_e_y_por_separado():
    class P:
        def __init__(self, x, y):
            self.x, self.y = x, y
    px = a_pixeles([P(0.5, 0.5)], ancho=640, alto=480)
    assert px.tolist() == [[320.0, 240.0]]


def test_postura_identidad_es_cero():
    assert postura_cabeza(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0))


def test_postura_detecta_cabeceo():
    a = np.radians(20)
    rx = np.array([[1, 0, 0, 0], [0, np.cos(a), -np.sin(a), 0],
                   [0, np.sin(a), np.cos(a), 0], [0, 0, 0, 1]])
    pitch, yaw, roll = postura_cabeza(rx)
    assert pitch == pytest.approx(20.0)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)
