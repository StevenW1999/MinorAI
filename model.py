import math
import data as data
import cv2
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ['anger', 'contempt', 'happy', 'sadness']


def get_landmarks(image):
    detections = detector(image, 1)
    landmarks_x = []
    landmarks_y = []
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
        landmarks_x = xlist
        landmarks_y = ylist


    return [landmarks_x, landmarks_y]


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect_face(img_path):
    if img_path == 'video':
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.flip(img, 1)  # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDet.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = gray[y:y + h, x:x + w]
                get_landmarks(img)
                cv2.putText(img, predict(img), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow('video', img)
            k = cv2.waitKey(1)
            if k == 27:  # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        frame = cv2.imread(img_path)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        else:
            facefeatures = ""

        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            if facefeatures == "":
                print("no face found in file: %s" % img_path)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = gray[y:y + h, x:x + w]  # Cut the frame to size

        return gray



def train(t_data):
    for f in t_data:
        try:
            face = detect_face(f)
            landmarks = get_landmarks(face)

            filename = f
            parts = filename.split('_')
            name = parts[0]
            parts2 = name.split('\\')
            emotion = parts2[1]

            new_x = [data.emotion_data[emotion][0][i] + landmarks[0][i] for i in range(len(data.emotion_data[emotion][0]))]
            new_y = [data.emotion_data[emotion][1][i] + landmarks[1][i] for i in range(len(data.emotion_data[emotion][1]))]

            new_x_average = [x / 2 for x in new_x]
            new_y_average = [y / 2 for y in new_y]

            data.emotion_data[emotion][0] = new_x_average
            data.emotion_data[emotion][1] = new_y_average
        except:
            print('no can do brother :')


def euclidean_distance(rowx1, rowx2, rowy1, rowy2):
    distance_x = 0.0
    distance_y = 0.0
    for i in range(len(rowx1)-1):
        distance_x += (rowx1[i] - rowx2[i])**2

    for i in range(len(rowy1) - 1):
        distance_y += (rowy1[i] - rowy2[i]) ** 2

    return math.sqrt(distance_x), math.sqrt(distance_y)


def predict(face):
    landmarks = get_landmarks(face)
    rowx = landmarks[0]
    rowy = landmarks[1]
    distances = []
    for emotion in emotions:
        try:
            distance = euclidean_distance(data.emotion_data[emotion][0], rowx, data.emotion_data[emotion][1], rowy)
            distances.append(distance)
        except:
            return 'no idea'

    min_val = min(distances)
    index = distances.index(min_val)
    return emotions[index]


def test(p_data):
    correct = 0
    incorrect = 0
    for f in p_data:
        face = detect_face(f)
        predict_face = predict(face)

        filename = f
        parts = filename.split('_')
        name = parts[0]
        parts2 = name.split('\\')
        emotion = parts2[1]

        if emotion == predict_face:
            correct += 1
        else:
            incorrect += 1
    accuracy = (1 / (correct + incorrect)) * correct
    print('correct: ' + str(correct) + '\n' + 'incorrect: ' + str(incorrect) + '\n' + 'accuracy: ' + str(accuracy))
