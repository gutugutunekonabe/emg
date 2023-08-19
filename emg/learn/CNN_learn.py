import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Reshape,Permute
from keras.optimizers import RMSprop
from keras.datasets import cifar10
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import os
import pickle
from PIL import Image
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras import backend
 

def list_imgs(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
def list_csv(directory, ext='csv'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def plot_history(history, 
                save_graph_img_path, 
                fig_size_width, 
                fig_size_height, 
                lim_font_size):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
   
    epochs = range(len(acc))

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size  
    plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_graph_img_path)
    plt.close()
    
def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def main():
    print("start")
    
#==========================変更項目===================================#
    # データ格納用のパス
    SAVE_DATA_DIR_PATH = "C:\\Users\\Alienware\\Desktop\\M-EMG\\"

#=====================================================================#

    print("File_Load")
    # なければ作成
    os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

    data_x = []
    data_y = []
    num_classes = 5

#############################################################################
#b
    print("Load_b")
    for filepath in list_csv("C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\conv\\学部\\v7c\\3.bot\\b"):
       List=np.loadtxt(filepath, delimiter=',')
       Listcut=np.array(List [:,:])
       pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
       img=img_to_array(pil_image)
       data_x.append(img)
       data_y.append(0) # 教師データ（正解）

#c
    print("Load_c")
    for filepath in list_csv("C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\conv\\学部\\v7c\\3.bot\\c"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(1) # 教師データ（正解）

#v
    print("Load_v")
    for filepath in list_csv("C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\conv\\学部\\v7c\\3.bot\\v"):
       List=np.loadtxt(filepath, delimiter=',')
       Listcut=np.array(List [:,:])
       pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
       img=img_to_array(pil_image)
       data_x.append(img)
       data_y.append(2) # 教師データ（正解）

#x
    print("Load_x")
    for filepath in list_csv("C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\conv\\学部\\v7c\\3.bot\\x"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(3) # 教師データ（正解）

#z
    print("Load_z")
    for filepath in list_csv("C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\conv\\学部\\v7c\\3.bot\\z"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(4) # 教師データ（正解） .


#####################################################################
        
    # NumPy配列に変換
    data_x = np.asarray(data_x)

    # 学習データはNumPy配列に変換し
    data_y = np.asarray(data_y)
    

    # 学習用データとテストデータに分割 stratifyの引数でラベルごとの偏りをなくす
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.15,stratify=data_y)

    # 学習データはfloat32型に変換し、正規化(0～1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], 500,4,1)
    x_test = x_test.reshape(x_test.shape[0],500, 4,1)

 

    # 正解ラベルをone hotエンコーディング
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print("ccc")
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')
    

    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(500,4,1),activation='relu'))
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.summary()
    
    epochs = 3
    batch_size = 10

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.15,
                    )
    
    acc = history.history['val_acc']
    loss = history.history['val_loss']

    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(loss)), loss, label='loss', color='blue')
    plt.plot(range(len(acc)), acc, label='acc', color='red')
    plt.xlabel('epochs')
    plt.show()

    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)
    
    #混同行列を表示
    from sklearn.metrics import confusion_matrix
    predict_classes = model.predict_classes(x_test)
    true_classes = np.argmax(y_test, 1)
    cmx = confusion_matrix(true_classes, predict_classes)
    print(cmx)
    #混同行列をヒートマップとして表示
    import seaborn as sns
    sns.heatmap(cmx, annot=True, fmt='g', square=True)
    plt.show()

    model_json_str = model.to_json()
    open(SAVE_DATA_DIR_PATH + 'model.json', 'w').write(model_json_str)
    model.save_weights(SAVE_DATA_DIR_PATH + 'weights.h5')

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "lawhistory.json", 'wb') as f:
        pickle.dump(history.history, f)
    
    

  


if __name__ == '__main__':
    main()