import numpy as np
import onnxruntime as ort
from numpy import random
from lib.utils_numpy import (letterbox, non_max_suppression, plot_one_box,
                             scale_coords)


class ObjectDetector():
    """
        YoLo v7 Object Detector
    """
    def __init__(self, weight_path, conf_thres=0.20, iou_thres=0.45, classes=None) -> None:
        # Model Config
        self.session = None
        self.input_cfg = None
        self.input_name = None
        self.save_weights_path = weight_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.stride = 32 # int(model.stride.max())  # model stride
        self.imgsz = 640
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # PostProcess Config
        self.conf_thres = conf_thres       # ObjectConfidenceThreshold -> 0.20
        self.iou_thres = iou_thres         # NMS için IOU Threshold -> 0.45 
        self.classes = classes             # Sınıf tespiti için filtre 0-80: int veya [0, 44]: array 
        self.agnostic_nms = False          # NMS için agnostic yöntemi

        # Sınıflar için rastgele renkler belirle
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.__StartNewSession()


    def __StartNewSession(self):
        self.session = ort.InferenceSession(self.save_weights_path, providers=self.providers)
        self.input_cfg = self.session.get_inputs()[0]
        self.input_name =  self.input_cfg.name


    def __Preprocess(self, frame):
        img = letterbox(frame, self.imgsz, stride=self.stride)[0]   # Yeniden boyutlandır.
        img = img[:, :, ::-1].transpose(2, 0, 1)    # BGR -> RGB: 3x416x416
        img = np.ascontiguousarray(img)             # Ram optimizasyonu
        img = np.array(img, dtype=np.float16)       # Half (float16) olarak inference için
        img /= 255.0                                # 0-1 Normalizasyon
        img = np.expand_dims(img, axis=0)           # 3x416x416 -> 1x3x416x416

        return img


    def __Inference(self, input):
        pred = self.session.run(output_names=None, input_feed={self.input_name : input})

        return pred[0]


    def __PostProcess(self, pred, img, canvas, verbose=False):
        """
            return: [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, class_number], canvas
        """
        detectedObjects = []

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # Batch olarak verilmişse çıkarımları tek tek yap
            s = ''

            if len(det):
                # Orijinal resim için koordinatları yeniden boyutlandır.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], canvas.shape).round()

                if verbose:
                    for c in np.unique(det[:, -1]):
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, " # Objenin isim ve puanını ekle
                        print(s)

                # Sonuçları canvas'a bastır.
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    detectedObjects.append([*xyxy, conf, cls])

        return detectedObjects, canvas


    def Detect(self, frame, draw=False):
        canvas = frame.copy()
        frame = self.__Preprocess(frame)
        pred = self.__Inference(frame)
        detectedObjects, canvas = self.__PostProcess(pred, frame, canvas)
        if draw: 
            canvas = self.__Draw2Canvas(canvas, detectedObjects)

        return detectedObjects, canvas


    def __Draw2Canvas(self, canvas, detected_objects):
        for item in detected_objects:
            print(item)
            xyxy = item[:4]
            score = item[4]
            oid = int(item[5])
            label = f'{self.names[oid]} - {score: .2f}'
            canvas = plot_one_box(xyxy, canvas, label=label, color=self.colors[oid])
        
        return canvas


