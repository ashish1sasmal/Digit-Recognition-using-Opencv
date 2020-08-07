import cv2
import numpy as np
import imutils
from imutils import contours
import sys

cv2.namedWindow("out",cv2.WINDOW_NORMAL)

labels = np.loadtxt('assets/digit_label.data',np.float32)
digits = np.loadtxt('assets/digits.data',np.float32)

model = cv2.ml.KNearest_create()
model.train(digits,cv2.ml.ROW_SAMPLE, labels)

img = cv2.imread("Tests/digits.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,100,220)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)

cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# cnts,_ = contours.sort_contours(cnts)
out = img.copy()

for (i,c) in enumerate(cnts):
    x,y,w,h = cv2.boundingRect(c)
    out = cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
    roi = gray[y:y+h,x:x+w]
    roismall = cv2.resize(roi,(10,10))
    roismall = roismall.reshape((1,100))
    roismall = np.float32(roismall)
    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
    string = str(int((results[0][0])))
    cv2.putText(out,string, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

cv2.imwrite("Results/result.png",out)
cv2.imshow('out',out)
cv2.waitKey(0)
