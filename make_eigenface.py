import cv2  # OpenCv 라이브러리 import
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

names = ["KRE", "KYW", "SHS", "LGE"]
team_name = 'HMH'

images=[]
labels=[]


for label, target_name in enumerate(names):
    for i in range(1, 101):
        default_image = cv2.imread(f'dataset/{team_name}_{target_name}/{team_name}_{target_name}_{i}.jpg')
        gray_image = cv2.cvtColor(default_image, cv2.COLOR_BGR2GRAY)
        face_detection = face_detector.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors=7, minSize=(120,120))
        for (x, y, w, h) in face_detection:
            face_img = gray_image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(label)  # 각 사람에 대한 레이블 할당: 0, 1, 2, 3
            

        flip_default_image = cv2.flip(default_image, 1) #좌우반전
        gray_image = cv2.cvtColor(flip_default_image, cv2.COLOR_BGR2GRAY)
        face_detection = face_detector.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors=7, minSize=(120,120))
        for (x, y, w, h) in face_detection:
            face_img = gray_image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = face_img / 255.0
            images.append(face_img.flatten())
            labels.append(label)  # 각 사람에 대한 레이블 할당: 0, 1, 2, 3
            

            # cv2.imshow("default_image", default_image)
            # cv2.imshow("face_img", face_img)
            # cv2.waitKey(0)

images = np.array(images)
labels = np.array(labels)

pca = PCA(n_components=0.95)  # 데이터의 95% 분산을 설명하는 주성분 개수 선택
pca.fit(images)
eigenfaces = pca.components_  # 주성분 벡터 (Eigenfaces)
# eigenfaces는 이미 pca.components_로 주성분 벡터가 저장됨

# PCA 그래프 그리기
def plot_pca_variance(pca):
    # 주성분의 분산 비율
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(len(explained_variance)), np.cumsum(explained_variance), where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title('PCA Explained Variance')
    plt.show()

# Eigenface 시각화
def plot_eigenfaces(eigenfaces, h, w, n_col=5, n_row=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(f"Eigenface {i + 1}")
        plt.xticks(())
        plt.yticks(())

    plt.show()

# PCA 분산 설명 그래프 그리기
plot_pca_variance(pca)

# Eigenface 시각화
plot_eigenfaces(eigenfaces, 200, 200)