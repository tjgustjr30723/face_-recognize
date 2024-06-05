import cv2  # OpenCv 라이브러리 import
import numpy as np
from sklearn.decomposition import PCA


face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')



names = ["PIW", "KHH", "CSB", "KDM", "LSE", 
         "PJS_team2", "KMG", "KMS_team2", "LJH", "KUS", 
         "KHY", "KMJ_team3", "KKJ", "KMS_team3", 
         "KRE", "KYW", "SHS", "LGE", 
         "CJY", "JUH", "MJY", "PSG", 
         "LSC", "RYJ", "LDW", "YSB", 
         "CYW", "JHS", "KMJ_team7", "PJS_team7", "YYS"]

# team_name = 'HMH'

images=[]
labels=[]
total_recog = []

for i, target_name in enumerate(names):
    recog_cnt = 0
    for j in range(1, 101):
        default_image = cv2.imread(f'dataset/{target_name}/{target_name}_{j}.jpg')
        gray_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2GRAY)
        face_detection = face_detector.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors=5, minSize=(120,120))
        for (x, y, w, h) in face_detection:
            recog_cnt += 1
            face_img = gray_image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(target_name)  # 각 사람에 대한 레이블 할당: 0, 1, 2, 3
            

        flip_default_image = cv2.flip(default_image, 1) #좌우반전
        gray_image = cv2.cvtColor(flip_default_image, cv2.COLOR_BGR2GRAY)
        face_detection = face_detector.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors=5, minSize=(120,120))
        for (x, y, w, h) in face_detection:
            recog_cnt += 1
            face_img = gray_image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(target_name)  # 각 사람에 대한 레이블 할당: 0, 1, 2, 3
    total_recog.append(recog_cnt)
    print(target_name)
            # cv2.imshow("default_image", default_image)
            # cv2.imshow("face_img", face_img)
            # cv2.waitKey(0)

images = np.array(images)
labels = np.array(labels)

# 얼굴인식확인 ###################################

for i, target_name in enumerate(names):
    print("name : %s, count : %d" %(target_name, total_recog[i]))

##############################################

pca = PCA(n_components=0.95)  # 데이터의 95% 분산을 설명하는 주성분 개수 선택
pca.fit(images)
eigenfaces = pca.components_  # 주성분 벡터 (Eigenfaces)
# eigenfaces는 이미 pca.components_로 주성분 벡터가 저장됨

# 새로운 얼굴 이미지 로드 및 전처리

def predict(img):
    main_img = img
    new_face = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    test_face_detection = face_detector.detectMultiScale(new_face, scaleFactor = 1.2, minNeighbors=5, minSize=(120,120))

    for (x, y, w, h) in test_face_detection:
        test_face = new_face[y:y+h, x:x+w]
        test_face = cv2.resize(test_face, (200, 200))
        test_face = test_face.flatten() / 255.0

        # 새로운 얼굴 이미지를 고유 얼굴 공간으로 투영
        transformed_face = pca.transform([test_face])

        # 기존 얼굴 데이터와의 유사성 측정 (유클리드 거리 등)
        distances = np.linalg.norm(transformed_face - pca.transform(images), axis=1)

        # 가장 유사한 얼굴 찾기
        closest_face_index = np.argmin(distances)
        closest_label = labels[closest_face_index]

        print(f"가장 유사한 얼굴의 레이블: {closest_label}")

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f'Label: {closest_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(img, f'Distances: {distances}', (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 0, 0), 2)
        # cv2.putText(color_image, f'Accuracy: {np.max(prediction) * 100:.2f}%', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', img)

while(True):
    cam = cv2.VideoCapture(0)
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    while(True):
        ret, img = cam.read()
        # 전체 이미지를 640x480으로 조정
        resized_img = cv2.resize(img, (640, 480))
        predict(resized_img)
            
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break




