# BareBones 🚀 by Lachlan Masson 
# A barebones implementation of the yoloV5 repository
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
    from yoloV5 import yoloV5.run
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from parameters import Parameters

# yoloV5 refactored into a class
class BareBones: 

    # initialising function
    def __init__(self, parameters):

        LOGGER.info("")

        self.source = str(parameters.source)
        self.save_img = not parameters.nosave and not  self.source.endswith('.txt')  # save inference images
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Directories
        self.save_dir = increment_path(Path(parameters.project) / parameters.name, exist_ok=parameters.exist_ok)  # increment run
        (self.save_dir / 'labels' if parameters.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        if parameters.logger:
            LOGGER.info("[BareBones] Initialising Model...")

        # Load model
        self.device = select_device(parameters.device)
        self.model = DetectMultiBackend(parameters.weights, device=self.device, dnn=parameters.dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine =  self.model.stride,  self.model.names,  self.model.pt,  self.model.jit,  self.model.onnx,  self.model.engine
        self.imgsz = check_img_size(parameters.imgsz, s=self.stride)  # check image size
   
        # Half
        parameters.half &= (self.pt or self.jit or self.engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

        if self.pt or self.jit:
            self.model.model.half() if parameters.half else self.model.model.float()

    # inferencing function
    def inference(self, parameters, imagePath):

        if parameters.logger:
            LOGGER.info("[BareBones] Inferencing...")

        # Dataloader
        cv_image = cv2.imread(imagePath)
        dataset = LoadImages(imagePath, cv_image, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        imgsz = [self.imgsz, self.imgsz]

        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz), half=parameters.half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
       
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if parameters.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(parameters.save_dir / Path(parameters.path).stem, mkdir=True) if parameters.visualize else False
            pred = self.model(im, augment=parameters.augment, visualize=parameters.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, parameters.conf_thres, parameters.iou_thres, parameters.classes, parameters.agnostic_nms, max_det=parameters.max_det)
            dt[2] += time_sync() - t3

            cones = []

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
              
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if parameters.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=parameters.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write and Store results
                    for *xyxy, conf, cls in reversed(det):

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        cones += xywh

                        if parameters.save_txt:  # Write to file
                            line = (cls, *xywh, conf) if parameters.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or parameters.save_crop or parameters.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if parameters.hide_labels else (self.names[c] if parameters.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if parameters.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if parameters.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(0)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image

        if parameters.logger:
            LOGGER.info(f'[BareBones] Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

        if parameters.save_txt or self.save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if parameters.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        if parameters.update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

        # Returning the results
        
        subArrayCount = len(cones)/4

        cones = np.array_split(cones, subArrayCount)

        return cones