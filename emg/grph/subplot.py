# 必要なモジュールをインポート
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pathlib import Path


savename="happy"
filepath = 'csv\\suzuki0208\\1.top\\e\\e0.csv'


#######################CSV処理のお話し############################################################
print(Path(filepath).read_text())##csvの読み取り
df = pd.read_csv(filepath, header=None)##縦軸がないのでないことを宣言（数字で置き換える）
names = ['CH1', 'CH2', 'CH3','CH4']
df = pd.read_csv(filepath, names=names)##横軸を決める
print(df)##処理後のCSVをターミナルに表示


######################サブプロットのお話し#########################################################
df.plot(subplots=True, layout=(4, 1),sharex=True, sharey=True, xlim=(0,100))

######################メインプロットのお話し#######################################################
#plt.plot(df, marker='.', markersize=0 )##マーカーの形状とサイズを指定(すべてのグラフがまとめて出る)


########################全体設定のお話し##########################################################
plt.xlabel("The number of data")
plt.ylabel("Myo value")

plt.savefig(savename)#savenameで決めた名前で保存
plt.show()##処理後のCSVをグラフで表示