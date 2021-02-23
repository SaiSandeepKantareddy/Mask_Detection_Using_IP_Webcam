# Import the required modules
import cv2
import time
import PIL.Image
from io import BytesIO
import numpy as np
import glob
import argparse
from math import pow,sqrt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import FPS
from threading import Thread
from imutils.video import FileVideoStream

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to our input image")
ap.add_argument("-c", "--confidence", type=float, help="confidence threshold")
args = vars(ap.parse_args())

labels = [line.strip() for line in open(r'C:\Users\sande\Downloads\mask-detector-master_68\mask-detector-master\class_labels.txt')]

# Generate random bounding box bounding_box_color for each label
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))
network = cv2.dnn.readNetFromCaffe(r'C:\Users\sande\Downloads\mask-detector-master_68\mask-detector-master\SSD_MobileNet_prototxt.txt', r'C:\Users\sande\Downloads\mask-detector-master_68\mask-detector-master\SSD_MobileNet.caffemodel')

# ----

# ### Detect faces on image using OpenCV
# Face detection with OpenCV and deep learning (Adrian)
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# load our serialized model from disk
caffe_model = 'deploy.prototxt.txt'
caffe_trained = 'res10_300x300_ssd_iter_140000.caffemodel'
caffe_confidence = 0.53
model_folder = r'./'
mask_model = "mask_mobile_net.h5"

if args["confidence"]:
    caffe_confidence = args["confidence"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(model_folder + caffe_model, 
                               model_folder + caffe_trained
                              )


model = load_model(model_folder + mask_model)


# Detect faces on image and call mask predictor
def detect_face_cnn(image, save = False, show = False):
    
    if image is not None:
        (h, w) = image.shape[:2]
        
        image_resized = cv2.resize(image, (300, 300))

        blob = cv2.dnn.blobFromImage(image_resized, 
                                     0.007843, (300, 300), 127.5)


        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()
        F = 615

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
           
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > caffe_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if class_id == 15.00:

                    # Draw bounding box for the object
                    cv2.rectangle(image, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                    label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                    #print("{}".format(label))


                    coordinates[i] = (startX, startY, endX, endY)

                    # Mid point of bounding box
                    x_mid = round((startX+endX)/2,4)
                    y_mid = round((startY+endY)/2,4)

                    height = round(endY-startY,4)

                    # Distance from camera based on triangle similarity
                    distance = (165 * F)/height
                    #print("Distance(cm):{dist}\n".format(dist=distance))

                    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm,y_mid_cm,distance)
                    

        # Distance between every object detected in a frame
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < 200:
                        close_objects.add(i)
                        close_objects.add(j)

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0,0,255)
            else:
                COLOR = (0,255,0)
            (startX, startY, endX, endY) = coordinates[i]

            cv2.rectangle(image, (startX, startY), (endX, endY), COLOR, 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # Convert cms to feet
            cv2.putText(image, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        if image is not None:
            (h, w) = image.shape[:2]
            
            image_resized = cv2.resize(image, (300, 300))

            blob = cv2.dnn.blobFromImage(image_resized, 
                                         1.0,
                                         (300, 300), 
                                         (104.0, 
                                          177.0, 
                                          123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]
               
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > caffe_confidence:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    try:
                        img_crop =   image[startY-10:endY+10, startX-10:endX+10]

                        # predict mask or not
                        pred, pred_res = predict_mask(img_crop)
                        
                        #print("Face Detection confidence:{:2f}".format(round(confidence,2)), pred)

                        label = "MASK" if pred_res == 0 else "NO-MASK"
                        color = (0,255,0) if pred_res == 0 else (0,0,255)

                        # cv2.putText(image, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                        # cv2.rectangle(image, (startX, startY), (endX, endY), color)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(image, (startX, startY), (endX, endY), color,2)
                        cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    except:
                        print("found crop errors {}".format(round(confidence,2)))

                
        if show:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return image
    else:
        print("image not found!")


# Predict if face is using mask or not
def predict_mask(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    # make predictions on the input image
    pred = model.predict(image)
    pred_res = pred.argmax(axis=1)[0]
    
    return pred, pred_res



def show_webcam():
	# fvs = FileVideoStream('rtsp://192.168.1.10:554').start()
	# time.sleep(1.0)
	fps = FPS().start()
	cam = cv2.VideoCapture('rtsp://192.168.1.10:554')
	#cam.set(cv2.CAP_PROP_BUFFERSIZE,1)
	#cam = cv2.VideoCapture(0)
	count=2
	i=np.zeros((1920,1080,3))
	while cam.isOpened():
	#while True:
	        try:
	            #t1 = time.time()
	            #cam=cam.get(cv2.CAP_PROP_BUFFERSIZE,3)
	            #frame=fvs.read()
	            # cam.set(3,640)
	            # cam.set(4,480)
	            # cam.set(cv2.CAP_PROP_FPS,5)
	            
	            
	            # ret, frame = cam.read()
	            # #print(frame.shape)
	            # i=np.append(i,frame).reshape(count,1920,1080,3)
	            # print(len(i))
	            # if len(i)>=100:
	            # 	for j in i:
        		ret,frame = cam.read()
        		#i.append(frame)
        		t1=time.time()
        		height , width , layers =  frame.shape
        		new_h=int(height/2)
        		new_w=int(width/2)
        		frame = cv2.resize(frame, (new_w, new_h))
        		frame = detect_face_cnn(frame)
        		cv2.imshow("Image", frame)
        		print('the time is:',time.time()-t1)
        		if cv2.waitKey(1) & 0xFF == ord('q'):
        			break
        		fps.update()
	            # count+=1
	        except KeyboardInterrupt:
	            print()
	            cam.release()
	            #fvs.stop()
	            print ("Stream stopped")
	            break



	fps.stop()
	#print('frame:',i)
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	cam.release()
	cv2.destroyAllWindows()
	#fvs.stop()


### MAIN AREA

# ### Check image source from file or Webcam

# select image or webcam
if args["image"] is not None:
    image = cv2.imread(args["image"])
    detect_face_cnn(image, show = True)
else:
    show_webcam()




