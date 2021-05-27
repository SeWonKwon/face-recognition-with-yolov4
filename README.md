# face-recognition-with-yolov4

목표: 커스텀 데이터 셋을 활용환 클라우드 환경에서 실시간 얼굴 분류
환경: 구글 코랩, yolov4, face_recognition

--------------------
# download weight:
* [weight]https://drive.google.com/u/0/uc?export=download&confirm=S6so&id=1t1IxhLkGlCf8rQy-OgmkrytBDVbHSmiZ

-------------------
# About weight

* Opendatset v6 : 4000장
* 직접촬영 : 3000장( 직접 활영은 Labeling tool : https://github.com/AlexeyAB/Yolo_mark 을 활용하여 레이블링 했습니다.)
* 직접 촬영분중 1/3 가량은 마스크를 쓴 상태의 사진임.
* 훈련 환경 : 구글 코랩 pro
* 훈련 횟수 : 6000회
* 인풋 데이타: 416\*416
* cfg 파일 다운로드 : https://drive.google.com/file/d/1-I7W3vTZVFzUbAnvHW6M0bnwiyIxWKZb/view?usp=sharing

![image](https://user-images.githubusercontent.com/73154316/119817799-5c6ace80-bf29-11eb-8c89-e437d452e9de.png)

--------------
# 파일 설명:

폴더: Easyup-Vongjur : 모델 구성부터 마지막 구현까지 상세하게 올려 놓은 강의 형식의 ipynb.

face_recognition_with_yolov4_in_colab.ipynb : 코랩에서 구현 할수 있게 짜여진 소스코드.

yolov4_added_by_face_recog_local.py : 로컬에서 활용할 수 있게 opencv를 사용하여 바꾼 소스코드. (위의 cfg파일과 weight 파일을 다운 받으셔서 사용 하시면 됩니다.)


---------------
# 모델 구성

1. yolov4 를 활용하여 커스텀 데이터 셋을 구성. (다양한 방법으로 시도 해 보았지만, 실제로 좋은 결과를 내지 못하고 결국에는 face 하나만 넣어서 모델을 구성하게 됨)
2. 구글 코랩을 활용하여 얼굴 인식 뿐만 아니라 분류도 하기 위해서 face_recognition 라이브러리를 활용.
3. 마스크 쓴 사진을 데이터 셋에 추가하여 데이터 셋 구성.
4. pickle 파일을 활용하여 face_recognition의 face_encdoing을 통하여 사람을 구분할 수 있게됨.
5. yolov4와 face_recognition의 차이점을 이용하여, **unkown, 특정인, 마스크 착용 미착용**을 구분하게 됨


---------------
# 발표영상 : https://www.youtube.com/watch?v=DV268uw_6xc

-------------------
# Reference :

**Core ML**
* https://github.com/AlexeyAB?tab=repositories
* https://gist.github.com/TakaoNarikawa
* https://github.com/Dodant
