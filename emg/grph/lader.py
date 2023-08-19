import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


#インライン宣言
#matplotlib inline

#レーダーチャート１
filepath = '２回目.csv'
NAME=('2times')
# print(Path(filepath).read_text())##csvの読み取り
df = pd.read_csv(filepath, header=None)##縦軸がないのでないことを宣言（数字で置き換える）
#names = ['CH1', 'CH2', 'CH3','CH4']
names = ['CH1', 'CH2', 'CH3','CH4','CH5', 'CH6', 'CH7','CH8','CH9']
df = pd.read_csv(filepath, names=names)##横軸を決める
# print(df)##処理後のCSVをターミナルに表示

df.loc['平均'] = df.mean()
print (df.tail(1))
A=df.tail(1)



def plot_polar(labels, values, imgname):
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    values = np.concatenate((values, [values[0]]))  # 閉じた多角形にする
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-')  # 外枠
    ax.fill(angles, values, alpha=1)  # 塗りつぶし
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  # 軸ラベル
    ax.set_rlim(0.0 ,50)
    ax.set_title(NAME, pad=10)
    fig.savefig(imgname)
 
    plt.close(fig)

labels = ['CH1', 'CH2', 'CH3','CH4','CH5', 'CH6', 'CH7','CH8','CH9']
values = df.mean()

plt.show()
plot_polar(labels, values, NAME+'.png')