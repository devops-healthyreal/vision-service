import os
import re
import numpy as np
import logging
from flask import request
from flask_restful import Resource
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image, ImageFilter

# 스케일링 요소 설정
scale_factor = 2
# x, y, rw, rh : 왼쪽 시작, 위쪽 시작, 자를 영역의 오른쪽 끝, 자를 영역의 하단 끝
section_ratio = (0.0, 0.3, 0.7, 0.6)

env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if env_path and not os.path.isabs(env_path):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_path)
raw_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_APPLICATION_CREDENTIALS_PATH = (
    os.path.abspath(raw_path) if raw_path else None
)
if GOOGLE_APPLICATION_CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS_PATH


# ---------------------------------------------------
# Google Vision Client 생성
# ---------------------------------------------------
def get_vision_client():
    if GOOGLE_APPLICATION_CREDENTIALS_PATH and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS_PATH):
        credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS_PATH)
        print(f"credentials: {GOOGLE_APPLICATION_CREDENTIALS_PATH}")
        print("✅ 구글 인증 JSON 경로 확인됨")
        client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        print("⚠️ GOOGLE_APPLICATION_CREDENTIALS 환경변수 확인 실패")
        client = vision.ImageAnnotatorClient()
    return client


# ---------------------------------------------------
# Flask Resource
# ---------------------------------------------------
class InOcr(Resource):
    def post(self):
        image_file = request.files.get('file')

        if not image_file:
            return {"error": "No file uploaded"}, 400

        logging.info('Received file: %s', image_file.filename)

        # PIL 이미지로 열기
        image = Image.open(image_file.stream)

        # Pillow 기반 전처리
        processed_image = make_scan_image(image, section_ratio, scale_factor)

        # OCR 수행
        filtered_num = detect_text(processed_image)

        return filtered_num


# ---------------------------------------------------
# Pillow 버전 이미지 전처리
# ---------------------------------------------------
def preprocess_image(image: Image.Image, scale_factor: float) -> Image.Image:
    """
    1. 흑백 변환
    2. 노이즈 제거 (MedianFilter)
    3. 스케일링 (리사이즈)
    """
    # 흑백 변환
    gray_image = image.convert("L")

    # 노이즈 제거 (Median Filter)
    # denoised_image = gray_image.filter(ImageFilter.MedianFilter(size=3))

    # 스케일링
    # #1248*1684,2160*3046
    width, height = gray_image.size
    new_size = (int(2160 * scale_factor), int(3046 * scale_factor))
    resized_image = gray_image.resize(new_size, Image.BICUBIC)

    return resized_image


# ---------------------------------------------------
# Pillow 버전 스캔 영역 자르기
# ---------------------------------------------------
def make_scan_image(image: Image.Image, section_ratio, scale_factor: float) -> Image.Image:
    """
    Pillow 이미지를 전처리 후 특정 영역만 잘라 반환
    """
    image = preprocess_image(image, scale_factor)

    width, height = image.size
    print(f"이미지 가로 길이: {width}")
    print(f"이미지 세로 길이: {height}")
    x, y, rw, rh = section_ratio
    crop_box = (
        int(width * x),
        int(height * y),
        int(width * rw),
        int(height * rh)
    )
    cropped = image.crop(crop_box)
    return cropped


# ---------------------------------------------------
# Google Vision OCR 호출
# ---------------------------------------------------
def detect_text(image: Image.Image):
    client = get_vision_client()

    # Pillow 이미지를 임시 파일로 저장
    temp_path = "temp.png"
    image.save(temp_path, format="PNG")

    with open(temp_path, "rb") as image_file:
        content = image_file.read()

    vision_image = vision.Image(content=content)
    response = client.text_detection(image=vision_image, image_context={"language_hints": ["ko"]})
    texts = response.text_annotations
    # print(f"모든 텍스트: {texts}")

    filtered_texts = []
    pattern = r'^\d*\.\d+$'
    total_height = 0
    total_width = 0
    cnt = 0
    for text in texts:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        width = max(v[0] for v in vertices) - min(v[0] for v in vertices)
        height = max(v[1] for v in vertices) - min(v[1] for v in vertices)
        # 조건에 맞는 영역만 OCR 결과 사용
        if width >= 120 and height >= 50 and re.fullmatch(pattern, text.description):
            num = float(text.description)
            if num >= 140:
                num -= 100
            print(f"텍스트 : {num}, width: {width}, height: {height}")
            # total_width += width
            # total_height += height
            # cnt = cnt+1
            filtered_texts.append((num, vertices[0][1], width, height))
            # print(f'"{text.description}" 좌표: {vertices}')
    # prior_height = int(total_height/cnt * 1.2)
    # prior_width = int(total_width/cnt * 1.4)
    # print(f"평균 height: {prior_height}")
    # print(f"평균 width: {prior_width}")
    # print(f"총 개수: {len(filtered_texts)}")

    # y 좌표 기준 정렬 후 텍스트만 반환
    filtered_texts = sorted(filtered_texts, key=lambda x: x[1])
    # if(len(filtered_texts) > 6):
    #     filtered_texts = [t for t, _, width, height in filtered_texts if width >= 120 and height >= 50]
    # else:
    filtered_texts = [t for t, _, _, _ in filtered_texts]

    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info: https://cloud.google.com/apis/design/errors"
        )

    return filtered_texts[0:5]
