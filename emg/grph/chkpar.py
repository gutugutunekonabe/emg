#ピアソンの積率相関係数を用いた筋電位の類似性の検証について

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

#names = ['CH1', 'CH2', 'CH3','CH4']


filepath = 'csv\\9.csv'
filepath1 = 'csv\\8.csv'


df = pd.read_csv(filepath, header=None)##縦軸がないのでないことを宣言（数字で置き換える）
df1 = pd.read_csv(filepath1, header=None)

df.loc['平均'] = df.mean()##CSV最後に平均の出力
df1.loc['平均'] = df1.mean()
print (df.tail(1))##最後の一行の読みとり
print (df1.tail(1))

EMG1 = df.mean()
EMG2 = df1.mean()

correlation, pvalue = stats.pearsonr(EMG1, EMG2)
print(correlation,pvalue)
############################################################################################
# #score[n]で平均を求めたい列を指定。
# md = df[0].mode()
# md1 = df[1].mode()
# md2 = df[2].mode()
# md3 = df[3].mode()
# md4 = df[4].mode()
# md5 = df[5].mode()
# md6 = df[6].mode()
# md7 = df[7].mode()


# md8 = df1[0].mode()
# md9 = df1[1].mode()
# md10 = df1[2].mode()
# md11 = df1[3].mode()
# md12 = df1[4].mode()
# md13 = df1[5].mode()
# md14 = df1[6].mode()
# md15 = df1[7].mode()


# df2 = (md[0],md1[0],md2[0],md3[0],md4[0],md5[0],md6[0],md7[0])
# print (df2)
# df3 = (md8[0],md9[0],md10[0],md11[0],md12[0],md13[0],md14[0],md15[0])
# print (df3)
# df2 = pd.DataFrame(columns=('0', '1', '2', '3', '4', '5', '6', '7'))
# df3 = pd.DataFrame(columns=('0', '1', '2', '3', '4', '5', '6', '7'))


# mode1 = df2.mean()
# mode2 = df3.mean()

# correlation, pvalue = stats.pearsonr(mode1, mode2)
# print(correlation,pvalue)