from ultralytics import YOLO
import os

"""
import zipfile
with zipfile.ZipFile(f"./Steel_defect_detection_V2.v1-v1.yolov12.zip","r") as zip_ref:
    zip_ref.extractall("data")
"""

# Load the model.
model = YOLO(f'./yolov12n.pt')


# Training.

results = model.train(
   data=os.path.abspath(f"./data/data.yaml"),
   imgsz=416,
   epochs=300,
   batch=32,
   name='yolov12n_300e'
)

val_result = model.val()

