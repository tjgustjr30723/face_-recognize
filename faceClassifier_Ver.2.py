import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 얼굴 이미지 데이터 로드 및 전처리
names = ["PIW", "KHH", "CSB", "KDM", "LSE", 
         "PJS_team2", "KMG", "KMS_team2", "LJH", "KUS", 
         "KHY", "KMJ_team3", "KKJ", "KMS_team3", 
         "KRE", "KYW", "SHS", "LGE", 
         "CJY", "JUH", "MJY", "PSG", 
         "LSC", "RYJ", "LDW", "YSB", 
         "CYW", "JHS", "KMJ_team7", "PJS_team7", "YYS"]

# face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

images = []
labels = []
total_recog = []

for i, target_name in enumerate(names):
    recog_cnt = 0
    for j in range(1, 101):
        default_image = cv2.imread(f'dataset/{target_name}/{target_name}_{j}.jpg')
        gray_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2GRAY)
        face_detection = face_detector.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))
        for (x, y, w, h) in face_detection:
            recog_cnt += 1
            face_img = gray_image[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(target_name)  # 각 사람에 대한 레이블 할당

        flip_default_image = cv2.flip(default_image, 1)  # 좌우반전
        gray_image = cv2.cvtColor(flip_default_image, cv2.COLOR_BGR2GRAY)
        face_detection = face_detector.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))
        for (x, y, w, h) in face_detection:
            recog_cnt += 1
            face_img = gray_image[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(target_name)  # 각 사람에 대한 레이블 할당
    total_recog.append(recog_cnt)
    print(target_name)

images = np.array(images)
labels = np.array(labels)

# Label Encoding: 문자열 레이블을 정수로 변환합니다.
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# PCA 적용: 데이터의 95% 분산을 설명하는 주성분 개수 선택
pca = PCA(n_components=0.95)
images_pca = pca.fit_transform(images)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels_encoded, test_size=0.3, random_state=42)

# 레이블을 원-핫 인코딩
num_classes = len(np.unique(labels_encoded))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 신경망 모델 정의
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(images_pca.shape[1],)))  # PCA로 축소된 입력 크기
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=1000, batch_size=100, validation_split=0.2)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# 새로운 이미지에 대한 얼굴 인식 및 예측 함수 정의
def predict_face(image_path):
    # 이미지 로드 및 전처리
    color_image = image_path
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    face_crop = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=6, minSize=(120, 120))
    for (x, y, w, h) in face_crop:
        face_img = gray_image[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))
        face_img = face_img / 255.0
        face_img_flatten = face_img.flatten()

        # PCA 변환
        face_img_pca = pca.transform([face_img_flatten])

        # 예측
        prediction = model.predict(face_img_pca)
        predicted_label = np.argmax(prediction, axis=1)

        print(f"Predicted label: {predicted_label[0]} (Confidence: {np.max(prediction) * 100:.2f}%)")

        # 결과 시각화
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(color_image, f'Label: {names[predicted_label[0]]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(color_image, f'Accuracy: {np.max(prediction) * 100:.2f}%', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', color_image)
        
        return predicted_label[0]

    # cv2.imshow('Face Recognition', color_image)
    return None

# 새로운 이미지 예측 테스트
for label, target_name in enumerate(names):
    while(True):
        cam = cv2.VideoCapture(0)
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")

        while(True):
            ret, img = cam.read()
            resized_img = cv2.resize(img, (640, 480))
            predict_face(resized_img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
