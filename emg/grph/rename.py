import glob
import os

path = 'kinden-data\\csv\\v9\\2.mid\\s\\'
files = glob.glob(path+'*')


for i, f in enumerate(files):
    fname = 's' +str(i) + '.csv'
    os.rename(f, path + fname)