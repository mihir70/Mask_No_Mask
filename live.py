import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
#l = []
cam = cv2.VideoCapture(0)
cam.set(4, 4800)  # set video widht
cam.set(4, 480)  # set video height
font = cv2.FONT_HERSHEY_SIMPLEX
count = 1
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
    )
    for(x, y, w, h) in faces:
        count += 1
        test = img[y:y+h, x:x+w]
        img1 = cv2.resize(test, (128, 128))
        test1 = []
        test1.append(np.array(img1))
        x_test = np.array([i for i in test1])
        x_test = x_test[0]/255.0
        x_test = x_test.reshape(1, 128, 128, 3)
        print(x_test.shape)
        model = load_model(r'C:\\Users\\HP\\Desktop\\iml\\mask_vs_nomask\\MobileV2')
        pred = model.predict(x_test)
        if pred[0][0]<=pred[0][1]:
            value='masked'
            prob=pred[0][1]
        else:
            value='NON masked'
            prob=pred[0][0]
        color = (0, 255, 0) if value == 'masked' else (0, 0, 255)

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0,225,0), 2)
        textoput = '{:s}-{:.2f}'.format(value, prob.max()*100)
        cv2.putText(img, textoput, (x-6, y-10), font, 0.70, color, 2)
        cv2.rectangle(img, (x-7, y-10), (x+w+7, y+h+12), color, 2)
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()
