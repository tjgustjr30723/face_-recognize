import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
#from google.colab.patches import cv2_imshow

# 얼굴 이미지 데이터 로드 및 전처리
names = ["KRE", "KYW", "SHS", "LGE"]
team_name = 'HMH'
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

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

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# 새로운 이미지에 대한 얼굴 인식 및 예측 함수 정의
def predict_face(image_path):
    # 이미지 로드 및 전처리
    color_image = image_path
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    face_crop = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in face_crop:
        face_img = gray_image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_img = face_img / 255.0
        face_img_flatten = face_img.flatten()

        # PCA 변환
        face_img_pca = pca.transform([face_img_flatten])

        # 예측
        prediction = model.predict(face_img_pca)
        predicted_label = np.argmax(prediction, axis=1)

        print(f"Predicted label: {predicted_label[0]} (Confidence: {np.max(prediction) * 100:.2f}%)")

        # 결과 시각화
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(color_image, f'Label: {names[predicted_label[0]]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', color_image)
        return predicted_label[0]
    
        

    print("얼굴을 찾지 못했습니다.")
    return None

# 새로운 이미지 예측 테스트
for label, target_name in enumerate(names):
    # for i in range(1, 100):
    #     predict_face(f'testset/HMH_KRE1_{i}.jpg')

    #     # color_image = cv2.imread(f'dataset/{team_name}_{target_name}/{team_name}_{target_name}_{i}.jpg')

    while(True):
        cam = cv2.VideoCapture(0)
        # face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

        # team_name = 'HMH'

        # For each person, enter one face initial
        # face_initial = 'LGE'
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")

        # Initialize individual sampling face count
        count = 0
        while(True):
            ret, img = cam.read()
            # print("asdf\n")
            # 전체 이미지를 640x480으로 조정
            resized_img = cv2.resize(img, (640, 480))
            faces = face_detector.detectMultiScale(resized_img, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(resized_img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1
                # 얼굴을 포함한 전체 조정된 이미지를 저장
                # cv2.imwrite(f'dataset/{team_name}_{face_initial}_{count}.jpg', resized_img)
                # cv2.imshow('image', resized_img)
            
            predict_face(resized_img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 100: # Take 100 face sample and stop video
                break

