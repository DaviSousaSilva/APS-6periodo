import subprocess
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    modelo = YOLO("yolo11n.pt")

    modelo.train(
        data="/home/davi/Downloads/Ferramentas-v4.v3i.yolov11/data.yaml",
        epochs=10,
        imgsz=416,
        batch=4,
        name="PIVC",
        device="cpu",
    )

    melhor = str(modelo.trainer.best)
    interface = Path(__file__).parent / "interface.py"
    subprocess.run([sys.executable, str(interface), melhor])


if __name__ == "__main__":
    main()