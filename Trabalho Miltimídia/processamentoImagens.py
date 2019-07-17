import numpy as np
import cv2
import mahotas

def write(img, texto, cor=(255,0,0)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (10,20), fonte, 0.5, cor, 0,
    cv2.LINE_AA)

imgColorida = cv2.imread('dados.jpeg')

img = cv2.cvtColor(imgColorida, cv2.COLOR_BGR2GRAY)

suave = cv2.blur(img, (7, 7))

T = mahotas.thresholding.otsu(suave)
bin = suave.copy()
bin[bin > T] = 255
bin[bin < 255] = 0
bin = cv2.bitwise_not(bin)

bordas = cv2.Canny(bin, 70, 150)

(objetos, lx) = cv2.findContours(bordas.copy(),
cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

write(img, "Tons de cinza", 0)
write(suave, "Suavizacao com Blur", 0)
write(bin, "Binarizacao com Metodo Otsu", 255)
write(bordas, "Detector de bordas Canny", 255)
temp = np.vstack([
np.hstack([img, suave]),
np.hstack([bin, bordas])
])
cv2.imshow("Quantidade de objetos: "+str(len(objetos)), temp)
cv2.waitKey(0)
imgC2 = imgColorida.copy()
cv2.imshow("Imagem Original", imgColorida)

cv2.drawContours(imgC2, objetos, -1, (255, 0, 0), 2)
write(imgC2, str(len(objetos))+" encontrados!")
cv2.imshow("Resultado", imgC2)
cv2.waitKey(0)

