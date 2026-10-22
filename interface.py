import tkinter as tk
from tkinter import filedialog
 
import cv2
from ultralytics import YOLO
 
modelo = YOLO("runs/detect/PIVC/weights/best.pt")
 
 
def carregar():
    caminhos = filedialog.askopenfilenames(
        title="Selecione as imagens",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")],
    )
    for caminho in caminhos:
        resultados = modelo.predict(caminho, conf=0.65)
        cv2.imshow("Resultado", resultados[0].plot())
        cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
janela = tk.Tk()
janela.title("PIVC")
 
texto = tk.Label(janela, text="Clique para carregar imagens")
texto.pack(pady=10)
 
botao = tk.Button(janela, text="Carregar imagens", command=carregar)
botao.pack(pady=10)
 
janela.mainloop()