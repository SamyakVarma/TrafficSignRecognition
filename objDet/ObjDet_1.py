from ultralytics import YOLO
import tensorflow as tf
import os

os.chdir('/mnt/d/code/Object Detection/OBJDETECTOR')
model=YOLO("yolov8m.pt")
model.train(data='descriptor.yaml',epochs=10)