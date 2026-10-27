import argparse
import subprocess
import sys
from pathlib import Path
 
from ultralytics import YOLO
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Caminho do data.yaml do dataset")
    args = parser.parse_args()
 
    modelo = YOLO("yolo11n.pt")
 
    modelo.train(
        data=args.data,
        epochs=50,
        imgsz=320,
        batch=4,
        name="PIVC",
        device="0",
    )
 
    modelo.val()
 
    melhor = str(modelo.trainer.best)
    interface = Path(__file__).parent / "interface.py"
    subprocess.run([sys.executable, str(interface), melhor])
 
 
if __name__ == "__main__":
    main()