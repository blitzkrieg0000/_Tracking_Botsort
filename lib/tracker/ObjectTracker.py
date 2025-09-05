from types import SimpleNamespace

import numpy as np
from numpy import random

from lib.tracker.mc_bot_sort import BoTSORT
from lib.utils_numpy import plot_one_box


class ObjectTracker():
    """
        BOTSORT Tracker    
    """
    def __init__(self) -> None:
        self.tracker = None
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.opt = SimpleNamespace()
        self.__DeployTracker()


    def __DeployTracker(self):
        #! TRACKER
        self.opt.name = "exp"                    # save results to project/name

        # BotSort
        self.opt.track_high_thresh = 0.3     # tracking confidence threshold
        self.opt.track_low_thresh = 0.05     # lowest detection threshold
        self.opt.new_track_thresh = 0.4      # new track thresh
        self.opt.track_buffer = 30           # the frames for keep lost tracks
        self.opt.match_thresh = 0.7          # matching threshold for tracking
        self.opt.aspect_ratio_thresh = 1.6   # threshold for filtering out boxes of which aspect ratio are above the given value.
        self.opt.min_box_area = 10           # filter out tiny boxes
        self.opt.mot20 = False               # fuse_score: fuse score and iou for association

        # CMC (camera compansate c++) 
        self.opt.cmc_method = "sparseOptFlow"    # cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc

        # ReID
        self.opt.with_reid = False   # with ReID module.
        self.opt.fast_reid_config = "fast_reid/configs/MOT17/sbs_S50.yml"    # reid config file path
        self.opt.fast_reid_weights = "pretrained/mot17_sbs_S50.pth"   # reid config file path
        self.opt.proximity_thresh = 0.5    # threshold for rejecting low overlap reid matches
        self.opt.appearance_thresh = 0.25    # threshold for rejecting low appearance similarity reid matches
        self.opt.jde = False
        self.opt.ablation = False

        self.tracker = BoTSORT(self.opt, frame_rate=30.0)


    def __Tracker(self, detected_objects, frame):
        if not isinstance(detected_objects, np.ndarray):
            detected_objects = np.array(detected_objects)
        online_targets = self.tracker.update(detected_objects, frame)
        online_results = []
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            tscore = t.score
            if tlwh[2] * tlwh[3] > self.opt.min_box_area:
                online_results.append([*tlbr, tscore, tcls, tid])

        return online_results


    def Track(self, detectedObjects, canvas, draw=False):
        trackedObjects = self.__Tracker(detectedObjects, canvas)
        if draw:
            canvas = self.__Draw2Canvas(canvas, trackedObjects)

        return trackedObjects, canvas


    def __Draw2Canvas(self, canvas, detected_objects):
        for item in detected_objects:
            xyxy = item[:4]
            score = item[4]
            oid = int(item[5])
            tid = int(item[6])
            label = f'{self.names[oid]} - {score: .2f} - {tid}'
            canvas = plot_one_box(xyxy, canvas, label=label, color=self.colors[oid])
        
        return canvas
        

