import cv2

from lib.detector.ObjectDetector import ObjectDetector
from lib.helper import CWD
from lib.tracker.ObjectTracker import ObjectTracker


if __name__ == '__main__':
    save_weights_path = "data/weight/yolov7/onnx/yolo7.onnx"
    objectDetector = ObjectDetector(save_weights_path, conf_thres=0.35, iou_thres=0.45, classes=[0])
    objectTracker = ObjectTracker()


    # cam = cv2.VideoCapture("http://192.168.0.12:8080/video")
    cam = cv2.VideoCapture("data/asset/crowd.mp4")
    

    while(True):
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        detectedObjects, canvas = objectDetector.Detect(frame)
        trackedObjects, canvas = objectTracker.Track(detectedObjects, canvas, draw=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow('', canvas)

