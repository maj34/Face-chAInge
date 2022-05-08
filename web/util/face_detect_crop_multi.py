from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name='weights', root='.'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                self.models[model.taskname] = model
            else:
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, threshold=-1, max_num=0):
        if threshold == -1:
            threshold = self.det_thresh
        bboxes, kpss = self.det_model.detect(img, threshold=threshold, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        
        ret = []
        align_img_list = []
        M_list = []
        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]
            M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
            align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
            align_img_list.append(align_img)
            M_list.append(M)

        return align_img_list, M_list, bboxes
