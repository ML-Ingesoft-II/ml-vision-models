"""Detector de microsueño (HU-09a) y de rostro no detectado (HU-09c).

Funciona por TIEMPO (marcas en segundos), no por número de cuadros, para que
dé el mismo resultado a 15 o a 30 cuadros por segundo. Esta misma lógica es la
que la app móvil debe reproducir; los casos de prueba en tests/ son el contrato.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

# Tipos de evento que se entregan al resto del sistema.
MICROSUENO = "MICROSUENO"
ROSTRO_NO_DETECTADO = "ROSTRO_NO_DETECTADO"
ROSTRO_RECUPERADO = "ROSTRO_RECUPERADO"


@dataclass
class Config:
    umbral_ear: float = 0.21          # por debajo, el ojo se considera cerrado (a validar en TT-03)
    duracion_microsueno_s: float = 1.5  # valor de la HU-09, a validar en TT-03
    sin_rostro_s: float = 5.0         # HU-09c
    ventana_perclos_s: float = 60.0   # para PERCLOS (insumo del estado "Precaución")


@dataclass
class Evento:
    tipo: str
    ts: float                 # segundos (marca del cuadro que dispara el evento)
    duracion_s: float         # cuánto llevaban los ojos cerrados / el rostro sin verse
    ear: float | None = None
    perclos: float | None = None
    pitch: float | None = None

    def a_dict(self) -> dict:
        return asdict(self)


@dataclass
class MicrosleepDetector:
    config: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        self._cerrado_desde: float | None = None
        self._microsueno_emitido = False
        self._ultimo_rostro: float | None = None
        self._sin_rostro_desde: float | None = None
        self._aviso_sin_rostro_emitido = False
        self._historial: deque[tuple[float, bool]] = deque()  # (ts, ojos_cerrados)

    # ------------------------------------------------------------------ API
    def actualizar(self, ts: float, ear: float | None, pitch: float | None = None) -> list[Evento]:
        """Procesa un cuadro. `ear=None` significa que no se detectó rostro en ese cuadro.

        Devuelve la lista de eventos nuevos (normalmente vacía).
        """
        if ear is None:
            return self._sin_rostro(ts)
        eventos = self._con_rostro(ts)
        cerrado = ear < self.config.umbral_ear
        self._registrar_perclos(ts, cerrado)

        if cerrado:
            if self._cerrado_desde is None:
                self._cerrado_desde = ts
            duracion = ts - self._cerrado_desde
            if duracion >= self.config.duracion_microsueno_s and not self._microsueno_emitido:
                self._microsueno_emitido = True  # un solo evento por cierre (CA1 de HU-09a)
                eventos.append(Evento(MICROSUENO, ts, round(duracion, 3), round(ear, 4),
                                      round(self.perclos(), 4), pitch))
        else:
            self._cerrado_desde = None
            self._microsueno_emitido = False
        return eventos

    def perclos(self) -> float:
        """Fracción del tiempo con ojos cerrados en la ventana reciente (0 a 1)."""
        if not self._historial:
            return 0.0
        return sum(1 for _, c in self._historial if c) / len(self._historial)

    # ------------------------------------------------------------ internos
    def _registrar_perclos(self, ts: float, cerrado: bool) -> None:
        self._historial.append((ts, cerrado))
        limite = ts - self.config.ventana_perclos_s
        while self._historial and self._historial[0][0] < limite:
            self._historial.popleft()

    def _sin_rostro(self, ts: float) -> list[Evento]:
        # Sin rostro no se puede saber si los ojos siguen cerrados: se reinicia el conteo.
        self._cerrado_desde = None
        self._microsueno_emitido = False
        if self._sin_rostro_desde is None:
            self._sin_rostro_desde = ts if self._ultimo_rostro is None else self._ultimo_rostro
        duracion = ts - self._sin_rostro_desde
        if duracion > self.config.sin_rostro_s and not self._aviso_sin_rostro_emitido:
            self._aviso_sin_rostro_emitido = True
            return [Evento(ROSTRO_NO_DETECTADO, ts, round(duracion, 3))]
        return []

    def _con_rostro(self, ts: float) -> list[Evento]:
        eventos: list[Evento] = []
        if self._aviso_sin_rostro_emitido and self._sin_rostro_desde is not None:
            eventos.append(Evento(ROSTRO_RECUPERADO, ts, round(ts - self._sin_rostro_desde, 3)))
        self._sin_rostro_desde = None
        self._aviso_sin_rostro_emitido = False
        self._ultimo_rostro = ts
        return eventos


def calibrar_umbral(ears_ojos_abiertos: list[float], factor: float = 0.75) -> float:
    """Umbral personal: un porcentaje de la mediana del EAR con los ojos abiertos.

    Pensado para la calibración silenciosa de los primeros ~30 s del viaje.
    """
    if not ears_ojos_abiertos:
        raise ValueError("Se necesita al menos una medida de EAR para calibrar")
    valores = sorted(ears_ojos_abiertos)
    mitad = len(valores) // 2
    mediana = valores[mitad] if len(valores) % 2 else (valores[mitad - 1] + valores[mitad]) / 2
    return mediana * factor
