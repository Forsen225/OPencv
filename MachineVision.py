import cv2
import numpy as np


# Создание бинароного изображения с использованием функции:
# cv2.CHAIN_APPROX_NONE позволяющая найти границы изображения и записать знгачения от
# начальной точки до конечной
#
"""
img = cv2.imread('C:\\Users\\Admin\\Desktop\\Ai_Opencv\\tiger.jpeg')

new_img = np.zeros(img.shape, dtype='uint8')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (5,5), 0) # размытие 

img = cv2.Canny(img, 100, 140) #контраст потчеркивания границ

con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # нахождение контура

cv2.drawContours(new_img, con, -1, (230, 111, 148), 1) # меням цвет координат линий границ

cv2.imshow('Proverka', new_img)
cv2.waitKey(0)
"""




# Рассматривается вид BRG,RGB,HCV,LUV
""" 
img = cv2.imread("C:\\Users\\Admin\\Desktop\\Ai_Opencv\\tiger.jpeg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(img) # Разделение на слои

img = cv2.merge([b,g, r]) # Слияние слоев
 
cv2.imshow("Proverca", img)
cv2.waitKey(0)
"""

"""
import numpy 

photo = cv2.imread("C:\\Users\\Admin\\Desktop\\Ai_Opencv\\tiger.jpeg")
img = numpy.zeros(photo.shape[:2], dtype = 'uint8') # создание кортежа uint8 - описание пикселя в диапазоне 0-255

circle=cv2.circle(img.copy(), (200, 300), 120, 255, -1) # закрашивание по контору 1, заливка -1, радиус 50
square = cv2.rectangle(img.copy(), (25, 25), (250, 350), 255, -1) # (25,25) - левая верхняя (250, 350) правая нижняя 

img = cv2.bitwise_and(photo, photo, mask=circle) # принимает два параметра и ищет одинаковые части и обьединает их 
#mask - маску можно использовать как предмет наложения 

#img = cv2.bitwise_or(circle, square) # соединяет и выводит всю деталь целиком
#img = cv2.bitwise_xor(circle, square) # соединяет и выводит все кроме наложенных обьектов 
#img = cv2.bitwise_not(circle) #инверсия(вырезание) круга
cv2.imshow("Proverka", img)
cv2.waitKey(0)
"""

# Подгрузка видео 
import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Track")
cv2.createTrackbar("T1", "Track", 0, 255, nothing)
cv2.createTrackbar("T2", "Track", 0, 255, nothing)

while True:
    success, img = cap.read()

    img = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh1 = cv2.getTrackbarPos("T1", "Track")
    thresh2 = cv2.getTrackbarPos("T2", "Track")

    canny = cv2.Canny(gray, thresh1, thresh2)

    faces = cv2.CascadeClassifier("C:\\Users\\Admin\\Desktop\\Ai_Opencv\\haarcascade_frontalface_default.xml")

    results = faces.detectMultiScale(gray, scaleFactor= 2, minNeighbors=3)

    for (x, y, w, h) in results:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)


        cv2.imshow("Detected Faces", img)
        cv2.imshow("Gray", gray)
        cv2.imshow("Canny", canny)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break


"""
# работа
import cv2

img = cv2.imread("C:\\Users\\Admin\\Desktop\\Ai_Opencv\\People.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier("C:\\Users\\Admin\\Desktop\\Ai_Opencv\\haarcascade_frontalface_default.xml")

results = faces.detectMultiScale(gray, scaleFactor= 2, minNeighbors=3)

for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)


cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
"""

