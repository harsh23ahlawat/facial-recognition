import os
import numpy as np
import streamlit as st
import cv2

st.title("Face Recognition Web App")

# Define the dataset path
dataset_path = "./data/"
os.makedirs(dataset_path, exist_ok=True)

# Function to load and prepare dataset
def load_dataset(dataset_path):
    faceData = []
    labels = []
    nameMap = {}
    classId = 0

    for f in os.listdir(dataset_path):
        if f.endswith(".npy"):
            nameMap[classId] = f[:-4]
            dataItem = np.load(os.path.join(dataset_path, f))
            m = dataItem.shape[0]
            faceData.append(dataItem)
            target = classId * np.ones((m,))
            labels.append(target)
            classId += 1

    if faceData and labels:
        XT = np.concatenate(faceData, axis=0)
        yT = np.concatenate(labels, axis=0).reshape((-1, 1))
        return XT, yT, nameMap
    else:
        st.warning("No data found to concatenate.")
        return None, None, None

# Function for KNN
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i], xt.flatten())
        dlist.append((d, y[i]))

    dlist = sorted(dlist)
    dlist = [item[1] for item in dlist[:k]]
    labels, cnts = np.unique(dlist, return_counts=True)
    idx = cnts.argmax()
    return int(labels[idx])

# Handle file upload
uploaded_file = st.file_uploader("Upload .npy file", type=["npy"])

if uploaded_file is not None:
    # Save the uploaded file to the dataset path
    file_path = os.path.join(dataset_path, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully")

# Load dataset
XT, yT, nameMap = load_dataset(dataset_path)

# Check if webcam works
if XT is not None and yT is not None:
    stframe = st.empty()
    run = st.checkbox('Run')
    cam = cv2.VideoCapture(0)
    model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    while run:
        success, img = cam.read()
        if not success:
            st.warning("Reading camera failed!")
            break

        faces = model.detectMultiScale(img, 1.3, 5)
        for x, y, w, h in faces:
            cropped_face = img[y:y + h, x:x + w]
            if cropped_face.size > 0:
                cropped_face = cv2.resize(cropped_face, (100, 100))
                classPredicted = knn(XT, yT, cropped_face.flatten())
                namePredicted = nameMap[classPredicted]
                cv2.putText(img, namePredicted, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        stframe.image(img, channels="BGR")

    cam.release()
