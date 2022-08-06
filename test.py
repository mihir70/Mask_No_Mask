from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
imge = cv2.imread('img1.jpg')
imge = cv2.resize(imge, (128, 128))
img = np.array(imge)/255.0
img = img.reshape(1,128,128,3)
print(img.shape)
model = load_model(r'C:\\Users\\HP\\Desktop\\iml\\mask_vs_nomask\\MobileV2')
#img=np.reshape(img,(128,128,3))
pred = model.predict(img)
print('Probability of being NON Masked: \n', pred[0][0])
print('Probability of being Masked: ', pred[0][1])
