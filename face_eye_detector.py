import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    print("XATO: Yuz modeli yuklanmadi!")
    exit()
if eye_cascade.empty():
    print("XATO: Ko‘z modeli yuklanmadi!")
    exit()

print("Modellar yuklandi. Kamera ishga tushmoqda...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("XATO: Kamera ochilmadi! USB kamerani ulang yoki 0 ni 1 qiling.")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera tasviri olinmadi!")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    yuz_soni = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "YUZ", (x, y - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )

        koz_soni = 0
        for (ex, ey, ew, eh) in eyes:
            # Ko‘zga ko‘k ramka
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(roi_color, "KO'Z", (ex, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            koz_soni += 1

        info = f"{koz_soni} ko'z"
        cv2.putText(frame, info, (x + w + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Yuz: {yuz_soni}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Real vaqtda: Yuz va Ko‘z aniqlash", frame)



cap.release()
cv2.destroyAllWindows()