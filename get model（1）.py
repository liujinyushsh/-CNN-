import random
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import joblib
import pickle

lv1=96
lv2=256
x=[]
y=[]
#读取图片并直接归一化
#此处将同一张图片多次读入，增大训练集数据量，即数据增强
for i in range(250):

    src=cv2.imread('pic/'+str(i)+'.jpg')
    src=cv2.Canny(src,lv1,lv2)
    src=np.array(src)
    src=src.astype(np.float32)/255
    x.append(src)
    a = 0.0#0.0类是stop类
    y.append(int(float(a)))
    x.append(src)
    a = 0.0#0.0类是stop类
    y.append(int(float(a)))

    if i<500:
        src2 = cv2.imread('pic2/' + str(i) + '.jpg')
        src2 = cv2.Canny(src2, lv1, lv2)
        #cv2.imshow(' ',src2)
        #cv2.waitKey(0)
        src2 = np.array(src2)
        src2 = src2.astype(np.float32) / 255
        x.append(src2)
        b = 1.0#1.0类是big类
        y.append(int(float(b)))
        x.append(src2)
        b = 1.0#1.0类是big类
        y.append(int(float(b)))

    src = cv2.imread('pic3/' + str(i) + '.jpg')
    src = cv2.Canny(src, lv1, lv2)
    src = np.array(src)
    src = src.astype(np.float32) / 255
    x.append(src)
    c =2.0  # 2.0类是sreach类
    y.append(int(float(c)))
    x.append(src)
    c =2.0  # 2.0类是sreach类
    y.append(int(float(c)))

    src = cv2.imread('pic4/' + str(i) + '.jpg')
    src = cv2.Canny(src, lv1, lv2)
    src = np.array(src)
    src = src.astype(np.float32) / 255
    x.append(src)
    d = 3.0  # 3.0类是setting类
    y.append(int(float(d)))
    x.append(src)
    d = 3.0  # 3.0类是setting类
    y.append(int(float(d)))

    src = cv2.imread('pic5/' + str(i) + '.jpg')
    src = cv2.Canny(src, lv1, lv2)
    src = np.array(src)
    src = src.astype(np.float32) / 255
    x.append(src)
    e = 4.0#4.0类是start类
    y.append(int(float(e)))
    x.append(src)
    e = 4.0#4.0类是start类
    y.append(int(float(e)))


    if i<125:
        src = cv2.imread('pic6/' + str(i) + '.jpg')
        src = cv2.Canny(src, lv1, lv2)
        src = np.array(src)
        src = src.astype(np.float32) / 255
        x.append(src)
        f = 5.0#5.0类是lose类
        y.append(int(float(f)))
        x.append(src)
        f = 5.0#5.0类是lose类
        y.append(int(float(f)))
        x.append(src)
        f = 5.0#5.0类是lose类
        y.append(int(float(f)))
        x.append(src)
        f = 5.0  # 5.0类是lose类
        y.append(int(float(f)))


x=np.array(x)
y=np.array(y)
y=y[:,None]
print(x[0].shape)
num=len(x)
z=[]
f=[]

shape=(96,96)
#获取有用的有效的信息，将格式错误的信息排除
for i in range(num-10):
    if x[i].shape==(120,120):
        x[i]=cv2.resize(x[i],shape)
        z.append(x[i])
        f.append(y[i])

#打乱图片数组顺序
for i in range(len(z)-1):
    ran=random.randint(0,len(z)-1)
    rannum=z[i]
    ranclass=f[i]
    f[i]=f[ran]
    z[i]=z[ran]
    z[ran]=rannum
    f[ran]=ranclass

x=np.array(z)
y=np.array(f)
num=len(x)

#分离出训练集与测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=0)

y_train_onehot=tf.keras.utils.to_categorical(y_train)
y_test_onehot=tf.keras.utils.to_categorical(y_test)

#模型构建，三个卷积层+三个池化层
model=tf.keras.Sequential([
    #输入层：定义输入大小，后面不需要再定义输入
    tf.keras.Input(shape=(96,96,1)),
    #卷积层：filters：卷积核的个数，代表了输出的通道数，为32时输出的图像为(n,n,32)；kernel：卷积核大小，一般为2,3,5；padding为same时，步长为一，且允许填充，故输出图大小与原图一样
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'),#当padding为valid时，不填充
    #池化层：pool_size:池化核尺寸；strides：移动步长，默认为池化核大小，即每个2*2(不重叠)的区域都取一个均值，得到图像大小为原来一半
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.Conv2D(filters=96,kernel_size=(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),
    #全连接层，300代表输出尺寸为(*,300)的数组
    tf.keras.layers.Dense(1000,activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    #此处6代表总类别数目
    tf.keras.layers.Dense(6,activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',#损失函数
    optimizer='adam',#优化器
    metrics=['accuracy']#评价函数
)

#fit函数中：batch_size为每次梯度下降时使用的样本数，越低训练次数越多，epoch是训练轮数，valid是分离出的测试集占比
#构建模型中的训练集数据
train_history=model.fit(x_train,y_train_onehot,batch_size=6,epochs=8,validation_split=0.2,verbose=1)

#测试模型的准确度
model.evaluate(x_test,y_test_onehot)

#若不存在则创建目录
dirs = 'testModel'
if not os.path.exists(dirs):
    os.makedirs(dirs)

# 保存模型
model.save('./testModel/model2.h5')# 保存模型 '*.h5'是保存的文件名
                                                                       # 当前最佳的模型为model.h5模型，训练时存入model2.h5模型
