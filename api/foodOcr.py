from flask_restful import Resource
from flask import request, make_response
import base64
import io
import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps


class FoodOcr(Resource):
    def __init__(self):
        self.session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
        self.names = ["갈비구이","갈치구이","고등어구이","곱창구이","닭갈비","더덕구이","떡갈비","불고기","삼겹살","장어구이","조개구이","조기구이","황태구이","훈제오리","계란국","떡국_만두국","무국","미역국","북엇국","시래기국","육개장","콩나물국","과메기","양념치킨","젓갈","콩자반","편육","피자","후라이드치킨","갓김치","깍두기","나박김치","무생채","배추김치","백김치","부추김치","열무김치","오이소박이","총각김치","파김치","가지볶음","고사리나물","미역줄기볶음","숙주나물","시금치나물","애호박볶음","경단","꿀떡","송편","만두","라면","막국수","물냉면","비빔냉면","수제비","열무국수","잔치국수","짜장면","짬뽕","쫄면","칼국수","콩국수","꽈리고추무침","도라지무침","도토리묵","잡채","콩나물무침","홍어무침","회무침","김밥","김치볶음밥","누룽지","비빔밥","새우볶음밥","알밥","유부초밥","잡곡밥","주먹밥","감자채볶음","건새우볶음","고추장진미채볶음","두부김치","떡볶이","라볶이","멸치볶음","소세지볶음","어묵볶음","오징어채볶음","제육볶음","주꾸미볶음","보쌈","수정과","식혜","간장게장","양념게장","깻잎장아찌","떡꼬치","감자전","계란말이","계란후라이","김치전","동그랑땡","생선전","파전","호박전","곱창전골","갈치조림","감자조림","고등어조림","꽁치조림","두부조림","땅콩조림","메추리알장조림","연근조림","우엉조림","장조림","코다리조림","전복죽","호박죽","김치찌개","닭계장","동태찌개","된장찌개","순두부찌개","갈비찜","계란찜","김치찜","꼬막찜","닭볶음탕","수육","순대","족발","찜닭","해물찜","갈비탕","감자탕","곰탕_설렁탕","매운탕","삼계탕","추어탕","고추튀김","새우튀김","오징어튀김","약과","약식","한과","멍게","산낙지","물회","육회"]

        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.img_size = 640

    # ---------------------------
    # 1️⃣ LetterBox 구현
    # ---------------------------
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # (h, w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        img = np.array(Image.fromarray(img).resize(new_unpad, Image.BICUBIC))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_pil = Image.fromarray(img)
        img_padded = ImageOps.expand(img_pil, border=(left, top, right, bottom), fill=tuple(color))
        img = np.array(img_padded)

        return img, r, (dw, dh)

    # ---------------------------
    # 2️⃣ NMS 구현
    # ---------------------------
    def nms(self, boxes, scores, iou_threshold=0.45):
        idxs = np.argsort(scores)[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            ious = self.iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_threshold]
        return np.array(keep)

    def iou(self, box, boxes):
        # x1,y1,x2,y2
        inter_x1 = np.maximum(box[0], boxes[:, 0])
        inter_y1 = np.maximum(box[1], boxes[:, 1])
        inter_x2 = np.minimum(box[2], boxes[:, 2])
        inter_y2 = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        return inter_area / np.maximum(union_area, 1e-6)

    # ---------------------------
    # 3️⃣ scale_boxes 구현
    # ---------------------------
    def scale_boxes(self, boxes, ratio, pad, orig_shape):
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
        return boxes

    def post(self):
        image_file = request.files.get("file")
        image_bytes = image_file.read()
        image_stream = io.BytesIO(image_bytes)
        im0 = Image.open(image_stream).convert("RGB")
        orig_w, orig_h = im0.size

        im = np.array(im0)
        im, ratio, pad = self.letterbox(im, (self.img_size, self.img_size))
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
        im = np.expand_dims(im, axis=0)

        input_name = self.session.get_inputs()[0].name
        preds = self.session.run(None, {input_name: im})[0][0]

        mask = preds[:, 4] > self.conf_thres
        preds = preds[mask]

        if preds.shape[0] == 0:
            return make_response(json.dumps({
                "base64": "",
                "detected_food_names": [],
                "message": "No objects detected"
            }, ensure_ascii=False))

        boxes = preds[:, :4]
        scores = preds[:, 4]
        classes = preds[:, 5].astype(int)

        # 좌표 변환 (cx,cy,w,h → x1,y1,x2,y2)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # NMS
        keep = self.nms(boxes_xyxy, scores, self.iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        classes = classes[keep]

        # 원본 크기로 복원
        boxes_xyxy = self.scale_boxes(boxes_xyxy, ratio, pad, (orig_h, orig_w))

        detected_food_names = [self.names[int(c)] for c in classes]

        # 시각화
        draw = ImageDraw.Draw(im0)
        for box, cls_id, conf in zip(boxes_xyxy, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.names[int(cls_id)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1 - 10), label, fill="green")

        # 저장 및 base64 인코딩
        save_dir = "./runs/detect/predict_onnx_nolite"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "new.jpg")
        im0.save(save_path)
        with open(save_path, "rb") as f:
            base64Predicted = base64.b64encode(f.read()).decode("utf-8")

        response_data = {
            "base64": base64Predicted,
            "detected_food_names": detected_food_names
        }
        return make_response(json.dumps(response_data, ensure_ascii=False))
