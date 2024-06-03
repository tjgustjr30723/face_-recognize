import cv2
import os

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

team_name = 'HMH'

# For each person, enter one face initial
face_initial = 'LGE'
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
        cv2.imwrite(f'dataset/{team_name}_{face_initial}_{count}.jpg', resized_img)
        cv2.imshow('image', resized_img)
        
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 100 face sample and stop video
        break