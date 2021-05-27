import cv2
import numpy as np
import pickle
import face_recognition as fr


CONFIDENCE = 0.9
THRESHOLD = 0.3
threshold = 0.38
LABELS = ['face']


cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromDarknet('cfg/yolov4-face.cfg', 'yolov4-face.weights')

with open('pickle/A.pickle', 'rb') as f:
    A_face_encodings = pickle.load(f)

with open('pickle/H.pickle', 'rb') as f:
    C_face_encodings = pickle.load(f)

GA_f_encoding = [A_face_encodings,C_face_encodings]
label_list = ['A','B'] # 이름 설정
colour_list = [(222,42,13),(13,222,56)]


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    H, W, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True)
    net.setInput(blob)
    output = net.forward()

    boxes, confidences, class_ids = [], [], []

    for det in output:
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            cx, cy, w, h = box * np.array([W, H, W, H])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
    print(idxs)


    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            # unknown = fr.face_encodings(img[y - 10:y+h + 10, x - 10:x+w + 10])
            unknown = fr.face_encodings(img[y :y + h , x :x + w ])
            print(unknown)
            if unknown == []:
                label2 = 'Take off your Mask'
                colour2 = (255, 255, 255)
                p = 0.00

            else:
                face_distance = fr.face_distance(GA_f_encoding, unknown[0])
                num = np.argmin(face_distance)
                confidence = face_distance[num]

                if confidence < 0.29:
                    label2 = label_list[num]
                    colour2 = colour_list[num]
                    p = 1.00

                elif confidence <= threshold:
                    label2 = label_list[num]
                    colour2 = colour_list[num]
                    p = 1 / ((0.2399 * confidence / (threshold - 0.3)) - (0.0720 / (threshold - 0.3)) + 1.01)

                else:
                    label2 = 'Unknown'
                    colour2 = (1, 1, 1)
                    p = 0.00
                    # continue

            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.putText(img, text='%s %.2f' % (label2, p), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colour2, thickness=2)


    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
