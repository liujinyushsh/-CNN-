import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import joblib
import pickle
import keras.models
import selectivesearch
import numpy as np
import cv2
import os
import tensorflow as tf
import keras


# 预测函数
def predict(img):
    model = keras.models.load_model('./testModel/model.h5')  # 加载模型
    #以上为测试集，下面为预测阶段
    xx=[]
    img=cv2.resize(img,(96,96))
    img = np.array(img)
    img = img.astype(np.float32) / 255
    xx.append(img)
    test=np.array(xx)

    predictions = model.predict(test)
    print(predictions)
    predictionsdex=np.argmax(predictions[0])
    print(predictionsdex)
    predictions=np.amax(predictions[0])
    print(predictions)
    return predictions,predictionsdex
# 非极大值抑制
def nms(classboxes,bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []
    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_class=[]
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_class.append(classboxes[index])
        a = start_x[index]
        b = order[:-1]
        c = start_x[order[:-1]]
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < threshold)
        order = order[left]
    return picked_boxes, picked_score,picked_class
# 输出图片
def cv_show(name, image):
    cv2.imwrite("./0.jpg", image)
    cv2.imshow(name, image)
    cv2.waitKey(0)

# 加载图片数据
img = cv2.imread('./test11.JPG')#selective_search()函数：将图像分割出完整的物体区域并调整尺寸便于输入分类器

img=cv2.resize(img,(0,0),fx=3.5,fy=3.5)
img_lbl, regions = selectivesearch.selective_search(img, scale=50, sigma=0.7, min_size=50)
# 计算一共分割了多少个原始候选区域
temp = set()
for i in range(img_lbl.shape[0]):
    for j in range(img_lbl.shape[1]):
        temp.add(img_lbl[i, j, 3])
# # 计算利用Selective Search算法得到了多少个候选区域
# 创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
candidates = set()
for r in regions:
    # 排除重复的候选区
    if r['rect'] in candidates:
        continue
    if r['size'] < 2000:
        continue
    # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
    candidates.add(r['rect'])
bounding_boxes = []
confidence_score = []
classboxes=[]
imOut = img.copy()
image = img.copy()
blur = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(blur, 30, 78)
cv_show(' ',img)
#获取分割区域并引入训练好的模型model
for x, y, w, h in candidates:
    im = img[y:y + h, x:x + w]
    if h*w<10000 and h/w<5 and h/w>0.2:
        preds,predsindex = predict(im)  # 预测结果
        if preds>0.25:
            #此处获取了标志的坐标，置信度，类别
            bounding_boxes.append((x, y, x + w, y + h))
            confidence_score.append(preds)
            classboxes.append(predsindex)
#经过非极大值抑制的标志坐标，置信度，类别
picked_boxes, picked_score,picked_class= nms(classboxes,bounding_boxes, confidence_score, 0.2)
for (start_x, start_y, end_x, end_y), confidence ,classes in zip(picked_boxes, picked_score,picked_class):
    #cv_show(" ",image[start_y:end_y,start_x:end_x])
    if classes==0:
        cv_show("stop", image[start_y:end_y, start_x:end_x])
    elif classes==1:
        cv_show("big", image[start_y:end_y, start_x:end_x])
    elif classes==2:
        cv_show("search", image[start_y:end_y, start_x:end_x])
    elif classes==3:
        cv_show("set", image[start_y:end_y, start_x:end_x])
    elif classes==4:
        cv_show("start", image[start_y:end_y, start_x:end_x])
    elif classes == 5:
        cv_show("lose", image[start_y:end_y, start_x:end_x])
    print('-'*60)
    print(confidence)
    print(classes)
    print((start_x,start_y,end_x,end_y))
print('-'*60)

for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_class):
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2, cv2.LINE_AA)
    if confidence==0:
        text='stop'
        cv2.putText(image, text, (start_x,start_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    elif confidence==1:
        text='big'
        cv2.putText(image, text, (start_x,start_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    if confidence==2:
        text='search'
        cv2.putText(image, text, (start_x,start_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    if confidence==3:
        text='set'
        cv2.putText(image, text, (start_x,start_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    if confidence==4:
        text='start'
        cv2.putText(image, text, (start_x,start_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    if confidence==5:
        text='lose'
        cv2.putText(image, text, (start_x,start_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
cv_show(' ',image)