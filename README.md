# ml-vision-models · Tire

Módulo de visión de Tire. Calcula medidas faciales con MediaPipe Face Landmarker y detecta microsueños (HU-09a) y la ausencia del rostro (HU-09c). Responsable: Julian.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

La primera vez que corres el demo se descarga el modelo `models/face_landmarker.task` (unos 4 MB).

## Uso

```bash
python scripts/demo_webcam.py               # webcam: muestra EAR, MAR, cabeceo y PERCLOS
python scripts/demo_webcam.py --calibrar    # umbral personal con los primeros 10 s
python scripts/demo_webcam.py --video data/raw/prueba.mp4 --csv data/processed/prueba.csv --sin-ventana
python -m pytest                            # casos de prueba (no necesitan cámara)
```

Los eventos salen por consola, uno por línea, en JSON:

```json
{"tipo": "MICROSUENO", "ts": 12.43, "duracion_s": 1.5, "ear": 0.12, "perclos": 0.08, "pitch": -14.2}
```

## Contrato con el resto del equipo

| Evento | Cuándo | Quién lo usa |
|---|---|---|
| `MICROSUENO` | Ojos cerrados ≥ 1,5 s; **uno solo** por cada cierre | Alarma en cabina (HU-10), escalamiento (HU-14) |
| `ROSTRO_NO_DETECTADO` | Más de 5 s sin rostro; uno solo por episodio | Aviso "No puedo ver tu rostro" (HU-09c) |
| `ROSTRO_RECUPERADO` | El rostro vuelve después del aviso; incluye la duración | Quitar el aviso y registrar la duración (HU-09c) |

- La app agrega `driver_id` y `trip_id`; este módulo no conoce al usuario.
- El detector trabaja por **tiempo**, no por cuadros: da lo mismo a 15 o a 30 fps.
- `tests/test_detector.py` es el contrato: la versión en React Native debe pasar los mismos casos.

### Parámetros (a validar en TT-03)

| Parámetro | Valor actual | Fuente |
|---|---|---|
| Umbral de EAR | 0,21 (o 75 % de la mediana personal con `--calibrar`) | Valor de partida usual en la literatura |
| Duración del microsueño | 1,5 s | HU-09 |
| Sin rostro | 5 s | HU-09 / HU-09c |
| Ventana de PERCLOS | 60 s | Insumo para el estado "Precaución" |

## Estructura

```
src/tire_ml/features.py   EAR, MAR, postura de la cabeza
src/tire_ml/detector.py   Detector de microsueño y de rostro ausente, calibración
scripts/demo_webcam.py    Prototipo con webcam o video; exporta CSV por cuadro
tests/                    Casos de prueba (contrato con móvil)
data/raw, data/processed  Videos y CSV (no se suben a Git; usar DVC o Drive)
models/                   Modelos (.task, .onnx, .tflite)
notebooks/                Análisis y validación del umbral
```

## Siguientes pasos

1. Probar el demo con la webcam y anotar el EAR con los ojos abiertos y cerrados, con y sin gafas.
2. Verificar los signos del cabeceo (`pitch`) bajando la cabeza.
3. Grabar videos del equipo (de día, de noche, con gafas, simulando cabeceos) y exportarlos a CSV con `--csv`.
4. En `notebooks/`, elegir el umbral y medir el recall y las falsas alarmas por hora (TT-03).
