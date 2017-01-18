#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from imutils import resize,exifrotate
import web
import json

urls = (
    '/.*', 'hello'
)

CLASSES = ('__background__',
	   'aeroplane', 'bicycle', 'bird', 'boat',
	   'bottle', 'bus', 'car', 'cat', 'chair',
	   'cow', 'diningtable', 'dog', 'horse',
	   'motorbike', 'person', 'pottedplant',
	   'sheep', 'sofa', 'train', 'tvmonitor')
	

def detect_core(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    exifrotate(im_file)
    im = cv2.imread(im_file)

    # timers
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, _t)
    timer.toc()
    print ('Detection took {:.3f}s for '
	   '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    all_detect_objects = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
	cls_ind += 1 # because we skipped background
	cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
	cls_scores = scores[:, cls_ind]
	dets = np.hstack((cls_boxes,
	                  cls_scores[:, np.newaxis])).astype(np.float32)
	keep = nms(dets, NMS_THRESH)
	dets = dets[keep, :]
	inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
	print ('There are {:d} {:s} SKU'.format(len(inds),cls))
	if len(inds) == 0:
	    continue
	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]
		detect_object = {
		    'class_name':cls,
		    'score':float(score),
		    'xmin':int(bbox[0]),
		    'ymin':int(bbox[1]),
		    'xmax':int(bbox[2]),
		    'ymax':int(bbox[3])
		}
		all_detect_objects.append(detect_object)

    return all_detect_objects
    
   
def api(path):
    timer = Timer()
    timer.tic()
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    #prototxt = os.path.join(cfg.MODELS_DIR,'..','pvanet','pva9.1','faster_rcnn_train_test_21cls.pt')
    prototxt = os.path.join(cfg.MODELS_DIR,'..','pvanet','pva9.1','faster_rcnn_train_test_ft_rcnn_only_plus_comp.pt')   
    #caffemodel = os.path.join(cfg.MODELS_DIR,'..','pvanet','pva9.1','PVA9.1_ImgNet_COCO_VOC0712.caffemodel')
    caffemodel = os.path.join(cfg.MODELS_DIR,'..','pvanet','pva9.1','PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel')
	                      

    if not os.path.isfile(caffemodel):
	raise IOError(('{:s} not found.\nDid you run ./data/script/'
	               'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # timers
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
	_, _= im_detect(net, im, _t)
   
    all_detect_objects = detect_core(net, path)

    timer.toc()

    data = {
        'object':all_detect_objects,
        'responsetime':timer.total_time
    }
   
    return json.dumps(data)

class hello:
    def GET(self):
        pic = web.input(path="no file path information")
        return api(pic.path)


#application = web.application(urls, globals()).wsgifunc()
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
