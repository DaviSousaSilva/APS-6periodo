from ultralytics import YOLO
 
 
def main():
    modelo = YOLO("yolo11n.pt")
 
    modelo.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="PIVC",
        device=0,
    )
 
 
if __name__ == "__main__":
    main()