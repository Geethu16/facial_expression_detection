import warnings
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


def model_architecture():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # number of classes are 7
    return model


def emotion_mapping_dictionary():
    _dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    return _dict


def get_predictions_from_image(image_path, path_to_saved_model, path_to_haar_cascade_face_detector):
    try:
        emotion_dict = emotion_mapping_dictionary()
        model = model_architecture()
        model.load_weights(path_to_saved_model)
        input_image = cv2.imread(image_path)
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(path_to_haar_cascade_face_detector)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            identified_facial_expression = emotion_dict[max_index]
            return identified_facial_expression
    except Exception as e:
        return print(f"Error in prediction identified as {e}")


if __name__ == "__main__":
    model_path = "model.h5"
    haar_cascade_face_feature_path = "haarcascade_frontalface_default.xml"
    input_image_path = "test_images\sample_image8.jpg"
    predicted_result = get_predictions_from_image(input_image_path, model_path, haar_cascade_face_feature_path)
    print("Identified facial expression is  ", predicted_result)
