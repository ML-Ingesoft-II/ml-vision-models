"""Casos de prueba de HU-09a y HU-09c. Son el contrato que la app móvil debe cumplir."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tire_ml.detector import (  # noqa: E402
    MICROSUENO, ROSTRO_NO_DETECTADO, ROSTRO_RECUPERADO, Config, MicrosleepDetector,
    calibrar_umbral,
)

FPS = 30
ABIERTO, CERRADO = 0.30, 0.10


def correr(detector, segmentos, fps=FPS, t0=0.0):
    """segmentos: lista de (valor_ear_o_None, segundos). Devuelve todos los eventos."""
    eventos, ts = [], t0
    for valor, segundos in segmentos:
        for _ in range(int(round(segundos * fps))):
            eventos += detector.actualizar(ts, valor)
            ts += 1 / fps
    return eventos


def tipos(eventos):
    return [e.tipo for e in eventos]


# --- HU-09a · CA1: un cierre de más de 1,5 s produce UN solo evento
def test_cierre_largo_emite_un_solo_microsueno():
    ev = correr(MicrosleepDetector(), [(ABIERTO, 1), (CERRADO, 3), (ABIERTO, 1)])
    assert tipos(ev) == [MICROSUENO]
    assert ev[0].duracion_s == pytest.approx(1.5, abs=1 / FPS)


# --- HU-09a · CA2: parpadeos normales no generan eventos
def test_parpadeos_normales_no_emiten():
    seg = [(ABIERTO, 2), (CERRADO, 0.3)] * 10 + [(ABIERTO, 1)]
    assert correr(MicrosleepDetector(), seg) == []


def test_cierre_justo_por_debajo_del_limite_no_emite():
    assert correr(MicrosleepDetector(), [(ABIERTO, 1), (CERRADO, 1.4), (ABIERTO, 1)]) == []


def test_dos_cierres_largos_separados_emiten_dos_eventos():
    ev = correr(MicrosleepDetector(), [(CERRADO, 2), (ABIERTO, 1), (CERRADO, 2)])
    assert tipos(ev) == [MICROSUENO, MICROSUENO]


# El resultado no debe depender de los cuadros por segundo del dispositivo
@pytest.mark.parametrize("fps", [10, 15, 30, 60])
def test_independiente_de_fps(fps):
    ev = correr(MicrosleepDetector(), [(ABIERTO, 1), (CERRADO, 2), (ABIERTO, 1)], fps=fps)
    assert tipos(ev) == [MICROSUENO]


# --- HU-09c · CA1 y CA3: más de 5 s sin rostro -> un aviso con su duración
def test_sin_rostro_mas_de_5s_emite_un_aviso():
    ev = correr(MicrosleepDetector(), [(ABIERTO, 1), (None, 7)])
    assert tipos(ev) == [ROSTRO_NO_DETECTADO]
    assert ev[0].duracion_s > 5.0


def test_sin_rostro_menos_de_5s_no_avisa():
    assert correr(MicrosleepDetector(), [(ABIERTO, 1), (None, 4), (ABIERTO, 1)]) == []


# --- HU-09c · CA2: al volver el rostro, se avisa que se recuperó y sigue el monitoreo
def test_rostro_recuperado_y_monitoreo_continua():
    ev = correr(MicrosleepDetector(), [(ABIERTO, 1), (None, 6), (ABIERTO, 1), (CERRADO, 2)])
    assert tipos(ev) == [ROSTRO_NO_DETECTADO, ROSTRO_RECUPERADO, MICROSUENO]


def test_perder_el_rostro_reinicia_el_conteo_de_ojos_cerrados():
    # 1 s cerrado + rostro perdido + 1 s cerrado no suman un microsueño
    assert correr(MicrosleepDetector(), [(CERRADO, 1), (None, 0.5), (CERRADO, 1)]) == []


def test_umbral_configurable():
    det = MicrosleepDetector(Config(umbral_ear=0.15))
    assert correr(det, [(0.18, 3)]) == []  # 0.18 > 0.15: se considera abierto


def test_perclos():
    det = MicrosleepDetector()
    correr(det, [(ABIERTO, 3), (CERRADO, 1)])
    assert det.perclos() == pytest.approx(0.25, abs=0.02)


def test_calibrar_umbral():
    assert calibrar_umbral([0.30, 0.28, 0.32]) == pytest.approx(0.30 * 0.75)
    with pytest.raises(ValueError):
        calibrar_umbral([])
