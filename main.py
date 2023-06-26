import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # зменшуємо розмір кадру
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # перетворюємо зображення у відтінки сірого
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Очікування натискання клавіші "q" для виходу з циклу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # знаходимо обличчя у відтінках сірого зображення
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # відображаємо обличчя на екрані
    for (x, y, w, h) in faces:
        x *= 2
        y *= 2
        w *= 2
        h *= 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # відображаємо відеопотік на екрані
    cv2.imshow('face', frame)

    # вихід з програми за натисканням клавіші "q"
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()