import numpy as np
import matplotlib.pyplot as plt


JES = "e"
                     
LOAD_DATA_DIR_PATH = "C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\csv\\suzuki0208\\1.top\\e"

SAVE_DATA_DIR_PATH = "C:\\Users\\Alienware\\Desktop\\emg-gpu\\kinden-data\\conv\\suzuki0208\\1.top\\e"



labels =['a', 'b', 'c', 'd','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk','ll','mm','nn','oo','pp','qq','rr','ss','tt','uu','vv','ww','xx','yy','zz',
'aaa','bbb','ccc','ddd','eee','fff','ggg','hhh','iii','jjj','kkk','lll','mmm','nnn','ooo','ppp','qqq','rrr','sss','ttt','uuu','vvv','www','xxx','yyy','zzz',
'aaaa','bbbb','cccc','dddd','ffff','gggg','hhhh','iiii','jjjj','kkkk','llll','mmmm','nnnn','oooo','pppp','qqqq','rrrr','ssss','tttt','uuuu','vvvv','wwww','xxxx','yyyy','zzzz']
yubi_labels =["q","w","e","r","t","y","a","s","d","f","g","z","x","c","v","b"]
for x in range(10):
    CSVList=np.loadtxt(LOAD_DATA_DIR_PATH+'\\'+JES+str(x)+'.csv', delimiter=',')



    for i in range(500,1000,1):

        img1 = np.array(CSVList [i-500:i,:],dtype = np.float)
        img  = img1
        np.savetxt(SAVE_DATA_DIR_PATH+"/"+str(i-500)+str(labels[x])+'.csv', img, delimiter=',', fmt='%d')
        print(i)


print("完了")