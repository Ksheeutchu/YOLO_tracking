import os
from dotenv import load_dotenv
from roboflow import Roboflow
load_dotenv()
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("rukas-lober-bk-ru").project("conveyor-part-detection")
version = project.version(2)
dataset = version.download("yolov8")
