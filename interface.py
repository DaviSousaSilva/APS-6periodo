import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
 
from PIL import Image, ImageTk
from ultralytics import YOLO
 
CONFIANCA = 0.4
 
 
def encontrar_modelo():
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidatos = sorted(
        Path("runs/detect").glob("PIVC*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidatos:
        sys.exit("Modelo não encontrado. Passe o caminho do best.pt como argumento.")
    return str(candidatos[-1])
 
 
modelo = YOLO(encontrar_modelo())
 
 
def mostrar_resultados(imagens):
    visor = tk.Toplevel(janela)
    visor.title("Resultado")
 
    fotos = [ImageTk.PhotoImage(img) for img in imagens]
    posicao = 0
 
    rotulo_imagem = tk.Label(visor)
    rotulo_imagem.pack()
    rotulo_posicao = tk.Label(visor)
    rotulo_posicao.pack()
 
    def atualizar():
        rotulo_imagem.config(image=fotos[posicao])
        rotulo_posicao.config(text=f"{posicao + 1}/{len(fotos)}")
 
    def mover(passo):
        nonlocal posicao
        posicao = (posicao + passo) % len(fotos)
        atualizar()
 
    barra = tk.Frame(visor)
    barra.pack(pady=5)
    tk.Button(barra, text="Anterior", command=lambda: mover(-1)).pack(side="left", padx=5)
    tk.Button(barra, text="Próxima", command=lambda: mover(1)).pack(side="left", padx=5)
 
    atualizar()
 
 
def carregar():
    caminhos = filedialog.askopenfilenames(
        title="Selecione as imagens",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")],
    )
    if not caminhos:
        return
 
    largura_max = int(janela.winfo_screenwidth() * 0.8)
    altura_max = int(janela.winfo_screenheight() * 0.8)
 
    imagens = []
    for caminho in caminhos:
        resultado = modelo.predict(caminho, conf=CONFIANCA)[0]
        imagem = Image.fromarray(resultado.plot()[:, :, ::-1])
        imagem.thumbnail((largura_max, altura_max))
        imagens.append(imagem)
 
    mostrar_resultados(imagens)
 
 
janela = tk.Tk()
janela.title("PIVC")
 
tk.Label(janela, text="Clique para carregar imagens").pack(pady=10)
tk.Button(janela, text="Carregar imagens", command=carregar).pack(pady=10)
 
janela.mainloop()