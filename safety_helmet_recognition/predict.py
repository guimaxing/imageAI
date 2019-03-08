# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:33:57 2019

@author: Moch
"""

# --coding:utf-8--
# 定义层
import sys
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
 
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input


target_size = (197, 197)
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)
 
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  print(preds)
  return preds[0]
 
# 画图函数
# 预测之后画图，这里默认是猫狗，当然可以修改label
 
labels = ('hat','no_hat')
def plot_preds(image, preds,labels):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()
 
# 载入模型
model = load_model('resnet50_model_weights.h5')
 
# 本地图片
img = Image.open('hat (366).jpg')
preds = predict(model, img, target_size)
print(preds)
#plot_preds(img, preds,labels)
