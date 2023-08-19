import matplotlib.pyplot as mplot
import numpy as np
targetData1=np.loadtxt('kindenDate\pattes1.csv',skiprows=1,delimiter=',',encoding="utf-8")
targetData2=np.loadtxt('kindenDate\pattes2.csv',skiprows=1,delimiter=',',encoding="utf-8")


print(targetData1.shape)
x=[]
y=[]
z=[]
for xyz in targetData1:
    x.append(xyz[0])
    y.append(xyz[1])
    z.append(xyz[2])

print(targetData2.shape)
x2=[]
y2=[]
z2=[]
for xyz2 in targetData2:
    x2.append(xyz2[0])
    y2.append(xyz2[1])
    z2.append(xyz2[2])

print(x)
print(y)
print(z)
print(x2)
print(y2)
print(z2)

fig=mplot.figure()
 
ax=fig.add_subplot(projection='3d')
ax.scatter(x,y,z,c='red',marker='o')
ax.scatter(x2,y2,z2,c='blue',marker='^')



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0,12)
ax.set_ylim(0,12)
ax.set_zlim(0,12)

mplot.show()