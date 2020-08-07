import cv2
import numpy as np
import imutils
from imutils import contours

cv2.namedWindow("norm",cv2.WINDOW_NORMAL)

img = cv2.imread("assets/digits.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray,100,220)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)
cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cnts,_ = contours.sort_contours(cnts)
out = img.copy()

labels = [0,9,8,7,6,5,4,3,2,1]

results =[]
datas = np.empty((0,100))
index=0

keys = [i for i in range(48,58)]

col = [(0,255,0), (0,0,255), (255,0,0)]

for (i,c) in enumerate(cnts):
    x,y,w,h = cv2.boundingRect(c)
    out = cv2.rectangle(out,(x,y),(x+w,y+h),col[i%3],2)
    roi = gray[y:y+h,x:x+w]
    roismall = cv2.resize(roi,(10,10))
    cv2.imshow('norm',out)
    key = cv2.waitKey(0)
    if key in keys:
        print(int(chr(key)))
        results.append(labels[int(chr(key))])
        data = roismall.reshape((1,100))
        datas = np.append(datas,data,0)
    elif key==27:
        break


results = np.array(results,np.float32)
results = results.reshape((results.size,1))

cv2.imshow("norm",out)
cv2.waitKey(0)
np.savetxt('assets/digit_label.data',results)
np.savetxt('assets/digits.data',datas)
