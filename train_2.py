import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from trafficsignnet_2 import TrafficSignNet,CosineDecay
import matplotlib.pyplot as plt
import skimage
from sklearn import metrics
import numpy as np
import argparse
import random
import os
#import cv2

SCALE=48

def load_split(basePath,csvPath):
    data=[]
    labels=[]

    rows=open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows) #row=(Width,Height,x1,y1,x2,y2,classID,ImgPath)
    for (i,row) in enumerate(rows):
        if i>0 and i%1000==0:
            print(f"[INFO] processed {i} images in total")
        
        (label,imagePath)=row.strip().split(",")[-2:]
        imagePath=os.path.sep.join([basePath,imagePath])
        image=skimage.io.imread(imagePath)
        #image=cv2.imread(imagePath)

        #resizeImg irrespective of aspect-ratio
        image=skimage.transform.resize(image,(SCALE,SCALE))
        #image=cv2.resize(image,(32,32))
        image=skimage.exposure.equalize_adapthist(image,clip_limit=0.1)
        #clahe=cv2.createCLAHE(clipLimit=0.1)
        #image=clahe.apply(image)

        data.append(image)
        labels.append(int(label))
    data=np.array(data)
    labels=np.array(labels)
    return(data,labels)

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="Path to input dataset")
ap.add_argument("-m","--model",required=False,help="Path to output model",default="./")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to training history plot")
args=vars(ap.parse_args())

NUM_EPOCHS=30
INIT_LR=0.02
BATCH_SIZE=32

labelNames=open("signnames.csv").read().strip().split("\n")[1:]
labelNames=[l.split(",")[1] for l in labelNames]

trainPath=os.path.sep.join([args["dataset"], "Train.csv"])
testPath=os.path.sep.join([args["dataset"],"Test.csv"])

print("[INFO] loading training and testing data...")
(trainX,trainY)=load_split(args["dataset"], trainPath)
(testX,testY)=load_split(args["dataset"], testPath)

trainX=trainX.astype("float32")/255.0
testX=testX.astype("float32")/255.0

numLabels=len(np.unique(trainY))
trainY=tf.keras.utils.to_categorical(trainY,numLabels)
testY=tf.keras.utils.to_categorical(testY,numLabels)

classTotals=trainY.sum(axis=0)
classWeight=dict()

for i in range(0,len(classTotals)):
    classWeight[i]=classTotals.max()/classTotals[i]

aug=tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split = 0.3,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
    vertical_flip=False
    )
SEED=42
tf.random.set_seed(42)

epochs = 30   # number of all epochs in training
stop_patience = 10   # number of epochs to wait before stopping training if monitored value does not improve
batches = int(len(trainX) / BATCH_SIZE)  # number of training batch to run per epoch

warmup_steps = batches * 5
learning_rate = CosineDecay(
    min_lr=1E-5, max_lr=1E-2, warmup_steps=warmup_steps
)

#earlystop = tf.keras.callbacks.EarlyStopping(
      #monitor='val_loss', patience=stop_patience, mode='min', restore_best_weights=True)


print("[INFO] compiling model...")

#opt=tf.keras.optimizers.Adam(learning_rate=INIT_LR,#weight_decay=INIT_LR/(NUM_EPOCHS*0.5))
                            #)
opt=tf.keras.optimizers.SGD(learning_rate=INIT_LR)
model=TrafficSignNet.build(width=SCALE,height=SCALE,depth=3,classes=numLabels)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
print("[INFO] training network...")

hist=model.fit(aug.flow(trainX,trainY,batch_size=BATCH_SIZE),
               validation_data=(testX,testY),
               steps_per_epoch=trainX.shape[0]//BATCH_SIZE,
               epochs=NUM_EPOCHS,
               #callbacks=earlystop,
               class_weight=classWeight,
               verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(metrics.classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, hist.history["loss"], label="train_loss")
plt.plot(N, hist.history["val_loss"], label="val_loss")
plt.plot(N, hist.history["accuracy"], label="train_acc")
plt.plot(N, hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
