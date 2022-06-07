import os
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelBinarizer

def load_label_data(data_path):

    labels = []

    with open(data_path) as f:
        rows = f.read().strip().split('\n')

        for row in rows:
            row = row.split(",")

            # extract only the label
            _, label, _, _, _, _ = row

            labels.append(label)

    labels = np.array(labels)

    # Return labels as numpy array
    return labels

def draw_result(frame, label, startX, startY, endX, endY):

    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]

    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 0), 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

    cv2.imshow('output', frame)

    return frame

def main():
    try:
        os.mkdir('./img')
    except OSError:
        pass

    data_path = 'baru.csv'

    print('=== Loading Label Data ... ===')

    labels = load_label_data(data_path)

    # One Hot Encoding The label
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print('==== Loading Label Data Success ===')

    model_path = 'detector.h5'
    print('=== Loading Model ... ===')
    
    model = load_model(model_path)
    print('==== Loading Model Success ===')

    video_path = './sample.mp4'
    output_path = './sample_output.mp4'

    print('=== Start Load Video ===')

    # Read video input
    cap = cv2.VideoCapture(video_path)
    print('FPS Input: '+str(cap.get(cv2.CAP_PROP_FPS)))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('Input Width: '+ str(width) + ', Input Height: ' + str(height))

    # Set Output writer
    # Args: Output Path, Codec, FPS, (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    idx = 0

    while True:
        # Get video frame
        idx+=1
        ret, frame = cap.read()

        if ret == True:

            # Preprocess Input
            input_img = cv2.resize(frame, (224,224))

            input_img = input_img / 255.0
            input_img = np.expand_dims(input_img, axis=0)

            # Predict
            (boxPreds, labelPreds) = model.predict(input_img)
            (startX, startY, endX, endY) = boxPreds[0]

            

            i = np.argmax(labelPreds, axis=1)
            label = lb.classes_[i][0]

            # Draw Bounding Box
            frame = draw_result(frame, label, startX, startY, endX, endY)

            if idx % 5 == 0:
                cv2.imwrite('./img/'+str(idx)+'.png', frame)
            
            # Save as mp4
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break
    
    cap.release()
    out.release()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()