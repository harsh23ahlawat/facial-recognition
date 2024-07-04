import cv2
import numpy as np
import os

dataset_path = "./data/"
faceData = []
labels = []
nameMap = {}
offset = 20
num_images_to_capture = 50

classId = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]

        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0]
        faceData.append(dataItem)

        target = classId * np.ones((m,))
        labels.append(target)  # Append target to labels list
        classId += 1

# Check if faceData and labels are not empty
if faceData and labels:
    XT = np.concatenate(faceData, axis=0)
    yT = np.concatenate(labels, axis=0).reshape((-1, 1))

    print(XT.shape)
    print(yT.shape)
    print(nameMap)
else:
    print("No data found to concatenate.")


def dist(p,q):
  return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt,k=5):

    m=X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d,y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:,1]

    labels,cnts = np.unique(labels,return_countd=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:
    success, img = cam.read()
    if not success:
        print("Reading camera failed!")
        break

    # Detect faces in the image
    faces = model.detectMultiScale(img, 1.3, 5)

    # Sort faces by size (area), largest first
    

    for f in faces:
        x, y, w, h = f
        

        # Ensure the cropping coordinates are within image bounds
        y_start = max(y - offset, 0)
        y_end = min(y + h + offset, img.shape[0])
        x_start = max(x - offset, 0)
        x_end = min(x + w + offset, img.shape[1])

        # Crop the face with some offset
        cropped_face = img[y_start:y_end, x_start:x_end]

        # Check if the cropped_face is not empty before resizing
        if cropped_face.size > 0:
            cropped_face = cv2.resize(cropped_face, (100, 100))

            
    # Display the image with rectangles drawn around faces
        

        classPredicted = knn(XT,yT,cropped_face.flatten())
        namePredicted = nameMap[classPredicted]
        cv2.putText(img,namePredicted, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    cv2.imshow("Prediction Window", img)
    # Break the loop if the desired number of images is reached
    if len(faceData) >= num_images_to_capture:
        print(f"Captured {num_images_to_capture} images. Exiting...")
        break

    cv2.waitKey(1)


cam.release()
cv2.destroyAllWindows()
