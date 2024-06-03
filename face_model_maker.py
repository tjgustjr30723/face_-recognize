import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 얼굴 이미지 데이터 로드 및 전처리
names = ["KRE", "KYW", "SHS", "LGE"]
team_name = 'HMH'
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

images = []
labels = []

for label, target_name in enumerate(names):
    for i in range(1, 101):
        color_image = cv2.imread(f'dataset/{team_name}_{target_name}/{team_name}_{target_name}_{i}.jpg')
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        face_crop = face_detector.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in face_crop:
            face_img = gray_image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(label)  # 각 사람에 대한 레이블 할당: 0, 1, 2, 3

# numpy 배열로 변환
images = np.array(images)
labels = np.array(labels)

# PCA 적용
pca = PCA(n_components=50)  # 주성분의 수 설정
images_pca = pca.fit_transform(images)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels, test_size=0.2, random_state=42)

# 레이블을 원-핫 인코딩
num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 신경망 모델 정의
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(50,)))  # PCA로 축소된 입력 크기
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 모델 저장
model.save('your_model.h5')
