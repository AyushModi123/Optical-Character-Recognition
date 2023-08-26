import easyocr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression
import imutils

reader = easyocr.Reader(['en','hi'], detector=True, recognizer = True)

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return (rects, confidences)

(W, H) = (None, None)
(newW, newH) = (320, 320)
(rW, rH) = (None, None)
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        _, frame = cam.read()
        start = time.time()
        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
        frame = cv2.resize(frame, (newW, newH))
        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        cv2.putText(orig,'Hold the text very still',(100, 30),font,1,(255,0,255),4)
        if key == ord('c'):
            for (startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                cropped = orig[startY:endY , startX:endX]
                text = reader.recognize(cropped, detail = 0, batch_size = 5)
                print(text)
                cv2.putText(orig,str(text),(startX, startY),font,1,(255,0,255),2)
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(orig,str(int(1/(time.time() - start))),(0,30),font,1,(255,0,255),2)
        cv2.imshow("Text Detection", orig)

        if key == ord("q"):
            break
except Exception as e:
    print(e)
cam.release()
cv2.destroyAllWindows()