import pandas as pd  #pandasモジュールをインポート
filepath = '1回目.csv'

#Datafreme型のデータを準備
df = pd.read_csv(filepath, header=None)##縦軸がないのでないことを宣言（数字で置き換える）





#pandas.mode関数を用いる。
#score[n]で平均を求めたい列を指定。
md = df[0].mode()
md1 = df[1].mode()
md2 = df[2].mode()
md3 = df[3].mode()
md4 = df[4].mode()
md5 = df[5].mode()
md6 = df[6].mode()
md7 = df[7].mode()



# #md[0]を指定することで最頻値のみ取得、指定しない場合はDataFreme型で返される
print("CH1 の最頻値は", md[0], "です。")
print("CH2 の最頻値は", md1[0], "です。")
print("CH3 の最頻値は", md2[0], "です。")
print("CH4 の最頻値は", md3[0], "です。")
print("CH5 の最頻値は", md4[0], "です。")
print("CH6 の最頻値は", md5[0], "です。")
print("CH7 の最頻値は", md6[0], "です。")
print("CH8 の最頻値は", md7[0], "です。")


