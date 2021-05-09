import cv2
img = cv2.imread('./pic/IMAGEnet.JPG')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('./result/imagenet.jpg',img)