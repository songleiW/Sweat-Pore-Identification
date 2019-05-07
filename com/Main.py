import numpy as np
from youtube_dl.extractor.medialaan import MedialaanIE
from plainbox.impl.secure import origin
import answer
from _cffi_backend import string
from _ast import If
from debian.debtags import output
from builtins import str
np.set_printoptions(threshold=np.nan)
import cv2
import matplotlib.pyplot as plt 
from PIL import Image
from matplotlib.pyplot import gray, subplot
import scipy.ndimage as ndi
from skimage import measure,color
import matplotlib.pyplot as plt
from skimage import data,segmentation,measure,morphology,color
from matplotlib.pyplot import figure
from numpy import uint8
from numpy import uint32
figure(num=None, figsize=(50, 50), dpi=80, facecolor='w', edgecolor='k')


#输入想要查看的图片编号
while True:
    figureNumber=None
    try:
        figureNumber=int(input("Please input the number(1-14) you want to view: "))
    except:
        pass
    if type(figureNumber)==int:
        if (figureNumber>0 and figureNumber <15):
            break
figureNumber=str(figureNumber)


#原始数据
figurePath='../figure/'
answerPath='../answer/figure '+figureNumber+' answer.txt'
file=open(answerPath,'w')
figurePath=figurePath+figureNumber+'.jpg'
originImg=cv2.imread(figurePath,0)#导入图片
plt.subplot(2,3,1)
plt.title('Origin Image '+figureNumber,fontsize=20)
plt.imshow(originImg,'gray')


#二值化
ret,binaryImage= cv2.threshold(originImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   #使用大津法二值化
binaryImage=binaryImage.astype(uint8)
plt.subplot(2,3,2)
plt.title('Binary Image',fontsize=20)
plt.imshow(binaryImage,'gray')


#使用联通分量除去闭汗孔，为下一步找角点做准备，并记录下闭汗孔的位置及大小
[rows, cols] = binaryImage.shape
boolImage=binaryImage.astype(bool)  
EliminaClosedImg=morphology.remove_small_objects(boolImage,min_size=60,connectivity=2) #联通区域小于60的视为闭汗孔，移除
EliminaClosedImg=EliminaClosedImg.astype(uint8)
plt.subplot(2,3,3)
plt.title('Elimination Closed Sweat Pore',fontsize=20)
plt.imshow(EliminaClosedImg,'gray')


#使用原始二值化图像减去消除了闭汗孔的二值化图片，获得仅有闭汗孔的图片，并标记出来
dst=cv2.subtract(binaryImage, EliminaClosedImg)
labels=measure.label(dst,connectivity=2)   #获得消除闭汗孔之后的联通分量
props=measure.regionprops(labels)   
plt.subplot(2,3,4)
plt.xlim(0,cols)
plt.ylim(rows,0)
plt.title('Maked Closed Sweat Pore',fontsize=20)
plt.imshow(binaryImage,'gray')
for regin in props:
    plt.scatter(regin.centroid[1].astype(uint32),regin.centroid[0].astype(uint32), color='', marker='o', edgecolors='y', s=40)#通过联通分量的质心位置做标记


#使用寻找角点的方法标记开汗孔
dst = cv2.cornerHarris(EliminaClosedImg,2,3,0.18) #三个参数：角点检测中要考虑的领域大小，求导中使用的窗口大小，角点检测方程中的自由参数
plt.subplot(2,3,5)
plt.title('Maked Open Sweat Pore',fontsize=20)
plt.imshow(EliminaClosedImg,'gray')
dst = cv2.dilate(dst,None)
openLocation=[] #记录下开汗孔的位置
openLocationResult=[] #记录下开汗孔的位置
for i in range(1,rows,4):
    for j in range(1,cols,4):
        if dst[i][j]>0.01*dst.max(): 
            openLocation.append([i,j])
for location1 in openLocation:
    flag=True
    for location2 in openLocationResult:
        if(abs(location1[0]-location2[0])<6 and abs(location1[1]-location2[1])<6):
            flag=False   #距离太近忽略
            break
    if flag:
        openLocationResult.append(location1)
        
for i in openLocationResult:
    plt.scatter(i[1],i[0], color='', marker='o', edgecolors='y', s=65)  #标注
    
    
#结果图
binaryImage=binaryImage.astype(uint8)
EliminaClosedImg=EliminaClosedImg.astype(uint8)
EliminaClosedImg=cv2.subtract(binaryImage, EliminaClosedImg)
labels=measure.label(EliminaClosedImg,connectivity=2)
props=measure.regionprops(labels)
plt.subplot(2,3,6)
plt.xlim(0,cols)
plt.ylim(rows,0)
plt.title('Final Result',fontsize=20)
plt.imshow(originImg,'gray')
openLocationResult.clear()

for regin in props:     #添加闭汗孔
    flag=True
    for location2 in openLocationResult:
        if (abs(regin.centroid[0]-location2[0])<6 and abs(regin.centroid[1]-location2[1])<6):
            flag=False   #距离太近的两个汗孔视为一个汗孔忽略
            break
    if flag:
        openLocationResult.append(regin.centroid)
        
for location1 in openLocation:
    flag=True
    for location2 in openLocationResult:
        if(abs(location1[0]-location2[0])<5 and abs(location1[1]-location2[1])<5):
            flag=False   #距离太近的两个汗孔视为一个汗孔忽略
            break
    if flag:
        openLocationResult.append(location1) 
        
for i in openLocationResult:    #标注汗孔
    plt.scatter(i[1],i[0], color='', marker='o', edgecolors='y', s=65)  #标注
    file.write(str(int(i[0]))+' '+str(int(i[1]))+'\n')       
plt.show()
file.flush()

