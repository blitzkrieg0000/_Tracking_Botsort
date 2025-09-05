import random
import time

import cv2
import numpy as np


def plot_one_box(bbox, canvas, color=None, label=None, line_thickness=1):
    tl = line_thickness or round(0.002 * (canvas.shape[0] + canvas.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(canvas, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        canvas = cv2.rectangle(canvas, c1, c2, color, -1, cv2.LINE_AA)  # filled
        canvas = cv2.putText(canvas, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return canvas


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)

    return coords


def clip_coords(boxes: np.array, img_shape):
    # Resimden dışarı taşan koordinatları yeniden boyutlandır.
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def xywh2xyxy(x):
    # xywh -> xyxy koordinatlarına çevir.
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
    """
        Non-Maximum Suppression (NMS)

        Return:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # [center_x, center_y, width, height, obj_confidence_score, class0, ..., class79] ya da
    xc = prediction[..., 4] > conf_thres    # Scorelar verilen eşik değerinden düşükse False yüksekse True olarak işaretle

    # Settings
    max_wh = 4096  # minumum ve maksimum kutu pixel boyutu (piksel)
    max_det = 300  # Her görüntüdeki maksimum tespit sayısı
    max_nms = 30000  # maksimum kutu boyutu
    time_limit = 10.0  # Timeout

    t = time.time()
    output = [ np.zeros([0, 6]) ] * prediction.shape[0]
    for xi, result in enumerate(prediction):  # image index, image inference
        result = result[xc[xi]]  # Eşik değerine uygun sonuçları filtrele

        # İşlenecek sonuç yoksa diğerine geç
        if not result.shape[0]:
            continue

        # Çok sınıflı sınıflandırmada ise (multiclass classification) eşik değerini obje eşik değeri ile çarpıyoruz.
        result[:, 5:] *= result[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(result[:, :4])

        # Detections matrix nx6 (xyxy, confidence_score, cls)
        conf = np.max(result[:, 5:], axis=1, keepdims=True)
        j = np.argmax(result[:, 5:], axis=1, keepdims=True)
        result = np.concatenate([box, conf, np.array(j, np.float32)], 1)#[conf > conf_thres]
        result = result[result[:, 4] > conf_thres]

        # Eğer sınıf filtresi varsa uygula. (Sadece bulunması istenen nesne id si: Filtrele)
        if classes is not None:
            result = result[np.any(result[:, 5:6] == classes, axis=1), :]

        # Boyut kontrolü yap
        n = result.shape[0]  # number of boxes
        if not n:  # Eğer hiç sonuç yoksa sonraki resulta geç
            continue
        elif n > max_nms:  # Bulunan sonuçlar istenilen nesne sayısını geçiyorsa fazlasını kırp; ignore la...
            result = result[result[..., 4].argsort(axis=0)[:max_nms]]

        #! Batched NMS
        # Burada obje niteliği taşıyan cisimlerin xyxy koordinatlarına sınıf_indexleri*4096 gibi bir sayı eklenerek,
        #iç içe geçmiş cisimlerin bbox kareleri ayrıştırılarak aynı cisim için hesaplanan olası kareler gruplanmış olur.
        #Ve NMS metodu uygulandığında bu cisimler daha iyi ayrıştırılırlar.
        c = result[:, 5:6] * (0 if agnostic else max_wh)  # class indexlerini max_wh ile çarp
        boxes, scores = result[:, :4] + c, result[:, 4]  # boxes (offset by class), scores
        
        #* NMS
        # CONF_THRESHOLD = 0.3
        # NMS_THRESHOLD = 0.4
        i = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        
        # Tespitleri sınırla
        if i.shape[0] > max_det:  
            i = i[:max_det]

        output[xi] = result[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Görüntüyü stride boyutuna göre yeniden boyutlandır ve doldur.
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Ölçek boyutu (yeni / eski): Yeniden boyutlandırma için gerekli
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # Sadece boyutunu düşür, eğer boyut büyütülmek isteniyorsa sınırla
        r = min(r, 1.0)

    # Padding'i ölç
    ratio = r, r  # width, height oranları
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minumum dikdörtgen
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # Resmi uzat
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # Resim dolgusunu her iki tarafa eşit dağıtmak için
    dh /= 2

    if shape[::-1] != new_unpad:  # Yeniden boyutlandır
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # Çerçeve ekle
    return img, ratio, (dw, dh)



