import cv2, sys, numpy, os
import boto3
from botocore.client import Config

#add your credentials here
ACCESS_KEY_ID = ''
ACCESS_SECRET_KEY = ''
BUCKET_NAME = ''

s3 = boto3.resource(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

size = 4
haar_file = 'D:/cloud-e-thon/haarcascade_frontalface_default.xml'
datasets = 'D:/cloud-e-thon/data'
# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Light Conditions...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
flag=1
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 90:

            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            flag=0
            frame1=im
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            frame2=im

    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:
        if flag==0:
            cv2.imwrite("./Logs/recognized.jpg",frame1)
            data=open("./Logs/recognized.jpg","rb")
            s3.Bucket("cloudethon").put_object(Key='Recognized/'+names[prediction[0]]+'/stream.jpg', Body=data)
        if flag!=0:
            cv2.imwrite("./Logs/not_recognized.jpg",frame2)
            data=open("./Logs/recognized.jpg","rb")
            s3.s3.Bucket("cloudethon").put_object(Key='Not-Recognized/stream.jpg', Body=data)
        break
