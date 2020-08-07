# Import the required modules
import cv2
import joblib
from skimage.feature import hog
import numpy as num
import pickle


with open("digits_cls.pkl", 'rb') as f:
    clf = pickle.load(f, encoding='latin1')

# Read the input image
image = cv2.imread("Tests/test8.png")

# Convert the image to Grayscale and then apply Gaussian filtering
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Threshold the image
ret, image_threshold = cv2.threshold(image_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
contrs, image_contrs = cv2.findContours(image_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rectangles = [cv2.boundingRect(ctr) for ctr in contrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rectangles :
    # Draw the rectangles
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    length = int(rect[3] * 1.6)
    point1 = int(rect[1] + rect[3] // 2 - length // 2)
    point2 = int(rect[0] + rect[2] // 2 - length // 2)
    roi = image_threshold[point1:point1+length, point2:point2+length]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualise=True)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_ARIEL, 2, (0, 255, 255), 3)

cv2.imshow("The Image is: ", image)
cv2.waitKey()
