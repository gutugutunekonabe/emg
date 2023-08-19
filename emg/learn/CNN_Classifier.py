import time
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import collections
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

import matplotlib.pyplot as plt


#==========================変更項目===================================#
sample_time = 15000 # 検証データ数[ms]



JES_labels =["q","w","e","r","t"]


#=====================================================================#

# ラベル
print("ok")
# 保存したモデル構造の読み込み
model = model_from_json(open("model.json", 'r').read())
print("ok")
# 保存した学習済みの重みを読み込み
model.load_weights("weights.h5")
print("モデルと重みの読み込み完了")



JES_COUNT = []
JES_pred = []
JES_true = []
sample = sample_time - 500


for x in range(len(JES_labels)):
    P = []
    JES = JES_labels[x]
    
    CSVList=np.loadtxt('C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\csv\\suzuki4\\test\\q1.csv', delimiter=',')
    JES_COUNT.append("----------------|"+JES+"|----------------")
    JES_COUNT.append("sample---"+ str(sample))
    
    for i in range(0,sample,1):
        pil_image = Image.fromarray(np.rot90(np.uint8(CSVList[i:500+i,:])))
        img=img_to_array(pil_image)
        img1 = img.astype('float32')/255.0
        img2 = np.array([img1])
        img3 = img2.reshape(img2.shape[0], 500,4,1)

        y_pred = model.predict(img3)
        # 最も確率の高い要素番号
        number_pred = np.argmax(y_pred) 
        print(JES,"_",i,"認識結果",JES_labels[int(number_pred)])

        JES_pred.append(JES_labels[int(number_pred)])
        P.append(JES_labels[int(number_pred)])

    for y in range(len(JES_labels)):
        JES_COUNT.append("[" + JES_labels[y]+ "]--------------" + "["+str(P.count(JES_labels[y])) +"]--------------" + "["+str((P.count(JES_labels[y]))/sample*100) + " %]")

    for j in range(sample):
        JES_true.append(JES_labels[x])


from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(JES_true, JES_pred)
print(cm)
#混同行列をヒートマップとして表示
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='g', square=True)
plt.show()

