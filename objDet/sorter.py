import os
import re
from PIL import Image
import numpy as np
import cv2
import random
import shutil
from pathlib import Path
ann_file="objDetector/dataset/train/gt.txt"
imgPath="objDetector/dataset/image"

# def parse_annotations(ann_file,img_dir):
#     annFile=open(ann_file)
#     imgNames=[]
#     boxes=[]
#     i=-1
#     for line in annFile:
#         imgName,x_min,y_min,x_max,y_max,classID=line.split(';')
#         imgName=imgName.split('.')[0]+'.png'
#         classID=re.sub('\D',"",classID)
#         if imgName in imgNames:
#             boxes[i].append([x_min,y_min,x_max,y_max,classID])
#         else:
#             i+=1
#             imgNames.append(img_dir+imgName)
#             boxes.append([])
#             boxes[i].append([x_min,y_min,x_max,y_max,classID])
#         # if w != IMAGE_W or h != IMAGE_H :
#         #     print('Image size error')
#         #     break
            
           
#     # Rectify annotations boxes : len -> max_annot
#     #imgNames = np.array(imgNames)
#     #boxes=np.asarray(boxes)
#     print(imgNames)
#     return imgNames, boxes


# parse_annotations(ann_file,imgPath)
# def converter(imgpath):
#     for img in os.listdir(imgpath):
#         impath=os.path.join(imgpath,img)
#         im=Image.open(impath)
#         imname=img.split('.')[0]
#         savepath=os.path.join("objDetector/dataset/image",imname+'.jpg')
#         im.save(savepath)

# imgpath="objDetector/dataset/train/image_ori"
# #converter(imgpath)
# img=cv2.imread("objDetector/dataset/train/image/00000.jpg")
# img-np.array(img)
# print(img.shape)

# def annotation_sorter(imgpath):
#     annFile=open(ann_file)
#     ImgNames=[]
#     for line in annFile:
#         imgName,x_min,y_min,x_max,y_max,classID=line.split(';')
#         imgName=imgName.split('.')[0]+'.jpg'
#         imagePath="objDetector/dataset/train/image/"+imgName    
#         imgName=imgName.split('.')[0]+'.txt'
#         imgName="objDetector/dataset/train/labels/"+imgName
#         img=cv2.imread(imagePath)
#         img=np.array(img)
#         imgWidth=img.shape[1]
#         imgHeight=img.shape[0]
#         if imgName in ImgNames:
#             newFile=open(imgName,'a')
#             k=1
#         else:
#             ImgNames.append(imgName)
#             newFile=open(imgName,'x')
#         classID=re.sub('\D',"",classID)
#         x_max=int(x_max)
#         x_min=int(x_min)
#         y_max=int(y_max)
#         y_min=int(y_min)
#         x_center=(x_min+x_max)/2
#         y_center=(y_min+y_max)/2
#         width=x_max-x_min
#         height=y_max-y_min
        
#         x_center=x_center/imgWidth
#         y_center=y_center/imgHeight
#         width=width/imgWidth
#         height=height/imgHeight

#         print(x_min,x_max,x_center,y_center)

#         newFile.write(f"{classID} {x_center} {y_center} {width} {height} \n")
#         print(f"wrote {imgName}")  
#         newFile.close()
# annotation_sorter(imgPath)

def splitter(ratio):
    random.seed(42)
    Num=599
    validationNum=int(Num*ratio)
    valSplit=[]
    valAnnotSplit=[]
    newpaths_imgs=[]
    newpaths_anns=[]
    while len(valSplit)<=validationNum:
        imgNum=random.randint(0,Num)
        if imgNum not in valSplit:
            valSplit.append(imgNum)
            valAnnotSplit.append(imgNum)
    newPath_img='objDetector/dataset/val/image/'  
    newPath_ann='objDetector/dataset/val/labels/'  
    for i,imgNum in enumerate(valSplit):
        if(imgNum<10):
            imgNum='0000'+str(imgNum)
        elif imgNum>=10 and imgNum<100:
            imgNum='000'+str(imgNum)
        elif imgNum>=100 and imgNum<1000:
            imgNum='00'+str(imgNum)
        valName=imgNum
        newpaths_imgs.append(newPath_img+imgNum+'.jpg')
        newpaths_anns.append(newPath_ann+imgNum+'.txt')
        imgNum='objDetector/dataset/train/image/'+imgNum+'.jpg'
        valName='objDetector/dataset/train/labels/'+valName+'.txt'
        valSplit[i]=imgNum
        valAnnotSplit[i]=valName
    for i in range(0,len(valSplit)):
        shutil.move(valSplit[i],newpaths_imgs[i])
        shutil.move(valAnnotSplit[i],newpaths_anns[i])


splitter(0.2)

# def emptyAnnotappender():
#     annFile=open(ann_file)
#     ImgNames=[]
#     for i in range(0,599):
#         if(i<10):
#             i='0000'+str(i)
#         elif i>=10 and i<100:
#             i='000'+str(i)
#         elif i>=100 and i<1000:
#             i='00'+str(i)
#         path='objDetector/dataset/train/labels/'+i+'.txt'
#         if Path(path).is_file():
#             continue
#         else:
#             file=open(path,'x')
#             file.close()
# emptyAnnotappender()

