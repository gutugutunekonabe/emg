# 正規分布の近似曲線


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

filepath = '1回目.csv'
number=0        #見たい列を入力する

df = pd.read_csv(filepath, header=None)##縦軸がないのでないことを宣言（数字で置き換える）
npd = df[number]
print (npd)

plt.hist(npd, bins=22, density=True)
mean, std = npd.mean(), npd.std()
xmin, xmax = npd.min(), npd.max()
x_axis = np.linspace(xmin, xmax, 1000)
pdf = norm.pdf(x_axis, mean, std)#正規分布曲線の表記



plt.plot(x_axis, pdf)
plt.show()