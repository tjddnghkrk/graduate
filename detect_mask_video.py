# USAGE

# 필요한 패키지
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys

def detect_and_predict_mask(frame, faceNet, maskNet):

        # 블롭 객체 생성
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

        # 네트워크 입력 설정하고 얼굴 인식
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # 배열 초기화
        faces = []
        locs = []
        preds = []

        # 사진에 얼굴이 여러개 있을 수 있으니 반복
        for i in range(0, detections.shape[2]):

                # 계산한 신뢰도 추출
                confidence = detections[0, 0, i, 2]

                # 설정한 최소한의 신뢰도를 충족할 때
                if confidence > args["confidence"]:

                        # 얼굴 영역 박스의 좌표를 구한다
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # 바운딩 박스가 프레임 위에 있도록 보장
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                        # 얼굴 추출하여 BGR을 RGB로 바꾼 후 resize, 전처리
                        face = frame[startY:endY, startX:endX]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(face, (224, 224))
                        face = img_to_array(face)
                        face = preprocess_input(face)

                        # 각각의 얼굴과 바운딩 박스를 넣는다
                        faces.append(face)
                        locs.append((startX, startY, endX, endY))

        # 얼굴이 감지된 경우에
        if len(faces) > 0:

                # 하나씩 마스크 detection을 진행
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)

        # 얼굴위치와 에측값을 넘긴다
        return (locs, preds)

# 인자 관리
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 얼굴 인식 모델을 가져온다
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# 마스크 인식 모델을 가져온다
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# 비디오 스트림을 시작한다
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 마스크 안낄 시의 처리를 위한 변수 및 이미지 로드
evaluation = 0
switch = 0
imgPath = "./image/maskpic.png"
image = cv2.imread(imgPath, cv2.IMREAD_ANYCOLOR)
image = cv2.resize(image, (700,500))

# 영상의 프레임을 가져온다
while True:

        #사이즈 맞추기
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        #얼굴을 인식하고 마스크 detect하는 함수
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # 반환된 얼굴 위치, 예측값을 순회하며
        for (box, pred) in zip(locs, preds):

                # 값들을 unpack하고
                (startX, startY, endX, endY) = box
                (mask, withoutMask, wrongmask) = pred # 3가지 경우로

                # label 결정하기 및 정확히 착용하지 않은 시간 카운트
                if mask > withoutMask and mask > wrongmask :
                        label = "Mask"
                        evaluation = 0

                elif withoutMask > wrongmask and withoutMask > mask :
                        label = "No Mask"
                        evaluation+=1

                else:
                        label = "Wrong Mask"
                        evaluation+=1

                # 라벨에 따라 초록, 주황, 빨간색으로 표기
                if label == "Mask" :
                        color = (0, 255, 0)
                if label == "Wrong Mask" :
                        color = (0, 125, 255)
                if label == "No Mask" :
                        color = (0, 0, 255)

                # 확률을 구한다
                result = "{}: {:.2f}%".format(label, max(mask, withoutMask, wrongmask) * 100)

                # 확률을 텍스트로 넣고 사각형 그리기
                cv2.putText(frame, result, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # 10초간 마스크 잘못쓰거나 안쓴다면
        if evaluation > 100 and switch == 0:
                cv2.namedWindow("WearMask")
                cv2.imshow("WearMask", image)
                cv2.waitKey(10000) # 이미지 띄우고 10초를 기다린다
                switch = 1

        # 기다린 10초가 지나면
        elif evaluation > 100:
                # PC방 시스템과 연결하여 로그아웃
                break

        # 기다리는 사이에 마스크를 끼면 이미지 닫고 다시 detect
        elif switch == 1 and label == "Mask":
                cv2.destroyWindow("WearMask")
                switch = 0

        # detect 하는 영상을 띄운다
        else:
                cv2.imshow("Detecting", frame)
                key = cv2.waitKey(1) & 0xFF

        # 'q'를 누르면 빠져나간다
        if key == ord("q"):
                break




# 창 닫고 프로그램 종료
cv2.destroyAllWindows()
vs.stop()
