# A class for passing parameters into detect.py

# libraries
import os
import sys
from pathlib import Path

class Parameters:

	def __init__(self, weights):
		
		FILE = Path(__file__).resolve()
		ROOT = FILE.parents[0]  # YOLOv5 root directory
		if str(ROOT) not in sys.path:
			sys.path.append(str(ROOT))  # add ROOT to PATH
		ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

		# init variables
		self.weights = weights 
		self.source = "" 

		# default variables 
		self.imgsz = 640 # this should be dynamic, sure there is a method to getting the size of an image
		self.conf_thres = 0.25
		self.iou_thres = 0.45
		self.max_det = 1000
		self.device = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
		self.view_img = True # show results
		self.save_txt = False # save results to *.txt
		self.save_conf = False  # save confidences in --save-txt labels
		self.save_crop = False # save cropped prediction boxes
		self.nosave = True # do not save images/videos
		self.classes = None # filter by class: --class 0, or --class 0 2 3
		self.agnostic_nms = False # class-agnostic NMS
		self.augment = False  # augmented inference
		self.visualize = False # augmented inference
		self.update = False # update all models
		self.project = ROOT / 'runs/detect'  # save results to project/name
		self.name = 'exp' # save results to project/name
		self.exist_ok = False,  # existing project/name ok, do not increment
		self.line_thickness = 3  # bounding box thickness (pixels)
		self.hide_labels = False  # hide labels
		self.hide_conf = False  # hide confidences
		self.half = False  # use FP16 half-precision inference
		self.dnn = False  # use OpenCV DNN for ONNX inference
		self.logger = False # logging toggle
