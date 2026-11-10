"""Prototipo HU-09a: webcam (o video) -> MediaPipe -> EAR/MAR/postura -> detector.

Uso:
    python scripts/demo_webcam.py                      # webcam 0
    python scripts/demo_webcam.py --calibrar           # umbral personal con los primeros 10 s
    python scripts/demo_webcam.py --video data/raw/x.mp4 --csv data/processed/x.csv
    python scripts/demo_webcam.py --sin-ventana --video ...   # sin interfaz (servidor/Colab)

Teclas: q para salir.
Los eventos salen por consola como JSON, una línea por evento.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tire_ml.detector import Config, MicrosleepDetector, calibrar_umbral  # noqa: E402
from tire_ml.features import a_pixeles, ear, mar, postura_cabeza  # noqa: E402

MODELO = RAIZ / "models" / "face_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")

ROJO, AMBAR, CIAN, BLANCO = (48, 59, 255), (11, 158, 245), (212, 182, 6), (252, 250, 248)


def asegurar_modelo() -> Path:
    if not MODELO.exists():
        print(f"Descargando el modelo de MediaPipe en {MODELO} ...")
        MODELO.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_MODELO, MODELO)
    return MODELO


def crear_landmarker() -> vision.FaceLandmarker:
    opciones = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(asegurar_modelo())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opciones)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="ruta de un video; si se omite, usa la webcam")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--csv", help="guardar las medidas por cuadro en este CSV")
    ap.add_argument("--umbral", type=float, default=Config.umbral_ear)
    ap.add_argument("--calibrar", action="store_true", help="calcular el umbral con los primeros 10 s")
    ap.add_argument("--sin-ventana", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camara)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la cámara o el video")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = MicrosleepDetector(Config(umbral_ear=args.umbral))
    landmarker = crear_landmarker()
    salida_csv = None
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        salida_csv = csv.writer(open(args.csv, "w", newline=""))
        salida_csv.writerow(["ts", "rostro", "ear", "mar", "pitch", "yaw", "roll",
                             "blink_izq", "blink_der", "jaw_open"])

    calibrando = args.calibrar
    muestras_calibracion: list[float] = []
    inicio = time.monotonic()
    n_cuadro, ultimo_evento, t_fps, fps = 0, "", time.monotonic(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_cuadro += 1
        # En video se usa el tiempo del archivo; en webcam, el reloj real.
        ts = n_cuadro / fps_video if args.video else time.monotonic() - inicio
        alto, ancho = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts * 1000))

        valor_ear = valor_mar = pitch = yaw = roll = None
        blend = {}
        if resultado.face_landmarks:
            puntos = a_pixeles(resultado.face_landmarks[0], ancho, alto)
            valor_ear, valor_mar = ear(puntos), mar(puntos)
            if resultado.facial_transformation_matrixes:
                pitch, yaw, roll = postura_cabeza(resultado.facial_transformation_matrixes[0])
            if resultado.face_blendshapes:
                blend = {c.category_name: c.score for c in resultado.face_blendshapes[0]}

        if calibrando:
            if valor_ear is not None:
                muestras_calibracion.append(valor_ear)
            if ts >= 10.0 and muestras_calibracion:
                detector.config.umbral_ear = calibrar_umbral(muestras_calibracion)
                calibrando = False
                print(f"Calibrado: umbral EAR = {detector.config.umbral_ear:.3f}")
        else:
            for ev in detector.actualizar(ts, valor_ear, pitch):
                print(json.dumps(ev.a_dict(), ensure_ascii=False), flush=True)
                ultimo_evento = ev.tipo

        if salida_csv:
            salida_csv.writerow([f"{ts:.3f}", int(valor_ear is not None),
                                 *(f"{v:.4f}" if v is not None else "" for v in
                                   (valor_ear, valor_mar, pitch, yaw, roll,
                                    blend.get("eyeBlinkLeft"), blend.get("eyeBlinkRight"),
                                    blend.get("jawOpen")))])

        if not args.sin_ventana:
            ahora = time.monotonic()
            if ahora - t_fps >= 1.0:
                fps, t_fps = n_cuadro / (ahora - inicio), ahora
            dibujar(frame, valor_ear, valor_mar, pitch, detector, calibrando, ultimo_evento, fps)
            cv2.imshow("Tire - prototipo HU-09a", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


def dibujar(frame, valor_ear, valor_mar, pitch, detector, calibrando, ultimo_evento, fps) -> None:
    y = 30
    def linea(texto, color=BLANCO):
        nonlocal y
        cv2.putText(frame, texto, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28

    if valor_ear is None:
        linea("Sin rostro", AMBAR)
    else:
        cerrado = valor_ear < detector.config.umbral_ear
        linea(f"EAR {valor_ear:.3f} (umbral {detector.config.umbral_ear:.3f})",
              ROJO if cerrado else BLANCO)
        linea(f"MAR {valor_mar:.3f}   pitch {pitch:+.1f}" if pitch is not None
              else f"MAR {valor_mar:.3f}")
    linea(f"PERCLOS {detector.perclos() * 100:.1f}%   {fps:.0f} fps", CIAN)
    if calibrando:
        linea("Calibrando: mira al frente con los ojos abiertos", CIAN)
    if ultimo_evento:
        linea(f"Ultimo evento: {ultimo_evento}", ROJO if ultimo_evento == "MICROSUENO" else AMBAR)


if __name__ == "__main__":
    main()
