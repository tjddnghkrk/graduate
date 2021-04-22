# USAGE
# python train_mask_detector.py --dataset dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 인자 관리
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# 학습률, epochs, batch size 설정
INIT_LR = 1e-4
EPOCHS = 30
BS = 32

# 이미지 경로 설정
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# 이미지 경로 내에서 반복
for imagePath in imagePaths:

	# 폴더명으로 class label 분류
	label = imagePath.split(os.path.sep)[-2]

	# 224x224로 이미지 가져오기
	image = load_img(imagePath, target_size=(224, 224))

	# 배열로 변환
	image = img_to_array(image)

	# 배치 생성을 위해 4차원 변환
	image = preprocess_input(image)

	# 이미지, 라벨 리스트 생성
	data.append(image)
	labels.append(label)

# numpy 배열로 변환
data = np.array(data, dtype="float32")
labels = np.array(labels)

print(labels)

# one-hot 인코딩
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(labels)
# 3개이기 때문에 to_categorical(labels) 아님

# 8:2 로 트레이닝 데이터와 테스트 데이터 분류
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 데이터 늘리기
aug = ImageDataGenerator(
	rotation_range=20, # 각도 돌리기
	zoom_range=0.1, # 0.9~1.1 확대
	width_shift_range=0.1, # 좌우 움직임
	height_shift_range=0.1, # 위아래 움직임
	shear_range=0.15, # 데이터 변형
	horizontal_flip=True, # 상하 뒤집기
	fill_mode="nearest")

# MobileNetV2 로드
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# headModel 작성
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# 모델 합치기
model = Model(inputs=baseModel.input, outputs=headModel)

# base 모델은 가중치 훈련 불가능하도록 만들기
for layer in baseModel.layers:
	layer.trainable = False

# 모델 컴파일 
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# head 학습
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# 테스트 데이터 예측
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# label index 찾기
predIdxs = np.argmax(predIdxs, axis=1)

# 분류 보고서
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# 모델 저장
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# 결과 plot으로 보여주기
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
