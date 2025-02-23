import os
import tempfile
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from keras.preprocessing.image import img_to_array

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'final_best_VGG16_weapon_detection_model.keras'

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = [folder for folder in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', folder))]

# Background subtractor for testing phase
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Function to preprocess a single image (grayscale)
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, IMAGE_SIZE)
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=-1)

# Function to preprocess frame
def preprocess_frame(frame, remove_background=False):
    if remove_background:
        fg_mask = bg_subtractor.apply(frame)
        frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
    
    processed_image = cv2.resize(frame, (224, 224))
    processed_image = img_to_array(processed_image)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image /= 255.0  
    return processed_image

# Function to detect weapons in a video
def detect_weapons_in_video(video_path, remove_background=False):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_image = preprocess_frame(frame, remove_background=remove_background)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label = f"{CLASS_NAMES[predicted_class]} ({confidence:.2f})"
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR", use_container_width=True)

    
    cap.release()

# Function to download video from Google Drive
def download_video_from_drive(file_id, credentials):
    drive_service = build('drive', 'v3', credentials=credentials)
    request = drive_service.files().get_media(fileId=file_id)
    fh = tempfile.NamedTemporaryFile(delete=False)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.name

# Streamlit UI
st.title("Weapon Detection in Videos")
st.write("Upload a video or provide a Google Drive link to detect weapons.")

SERVICE_ACCOUNT_FILE = "weapon-detection-451413-26c91c7aecdb.json"
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
drive_link = st.text_input("Or enter Google Drive file ID:")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    detect_weapons_in_video(video_path, remove_background=True)  # Remove background in testing

elif drive_link:
    video_path = download_video_from_drive(drive_link, credentials)
    detect_weapons_in_video(video_path, remove_background=True)  # Remove background in testing

# import os
# import tempfile
# import streamlit as st
# import tensorflow as tf
# import cv2
# import numpy as np
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# from keras.preprocessing.image import img_to_array


# # Constants
# IMAGE_SIZE = (224, 224)
# MODEL_PATH = 'final_best_VGG16_weapon_detection_model.keras'  # Change to ResNet50 if needed

# # Load the trained model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Define class names (replace with your actual class names)
# CLASS_NAMES = [folder for folder in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', folder))]
#  # Update with your actual classes
 

# # Function to preprocess a single image (grayscale)
# def preprocess_image(image):
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Resize the image to the target size
#     resized_image = cv2.resize(gray_image, IMAGE_SIZE)
    
#     # Normalize and expand dimensions for model prediction
#     normalized_image = resized_image / 255.0
#     return np.expand_dims(normalized_image, axis=-1)  # Add channel dimension for grayscale

# # Function to preprocess frame (for grayscale and RGB handling)
# def preprocess_frame(frame):
#     processed_image = cv2.resize(frame, (224, 224))  # Resize frame to (224, 224)

#     # Convert grayscale to RGB if needed (assuming grayscale input)
#     if len(processed_image.shape) == 3 and processed_image.shape[-1] == 1:
#         processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

#     processed_image = img_to_array(processed_image)  # Convert to array
#     processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
#     processed_image /= 255.0  # Normalize to [0, 1]
    
#     return processed_image

# # Function to detect weapons in a video
# def detect_weapons_in_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     stframe = st.empty()  # Streamlit placeholder for video frames

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocess the frame (grayscale)
#         processed_image = preprocess_frame(frame)

#         # Make prediction
#         predictions = model.predict(processed_image)
#         predicted_class = np.argmax(predictions[0])  # Get the class with the highest probability
#         confidence = np.max(predictions[0])  # Confidence of the prediction
#         label = f"{CLASS_NAMES[predicted_class]} ({confidence:.2f})"  # Label with confidence

#         # Display the label on the frame
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Display the frame in Streamlit
#         stframe.image(frame, channels="BGR", use_column_width=True)

#     cap.release()

# # Function to download video from Google Drive
# def download_video_from_drive(file_id, credentials):
#     drive_service = build('drive', 'v3', credentials=credentials)
#     request = drive_service.files().get_media(fileId=file_id)
#     fh = tempfile.NamedTemporaryFile(delete=False)
#     downloader = MediaIoBaseDownload(fh, request)
#     done = False
#     while done is False:
#         status, done = downloader.next_chunk()
#     return fh.name

# # Streamlit UI
# st.title("Weapon Detection in Videos")
# st.write("Upload a video or provide a Google Drive link to detect weapons.")

# # Load service account credentials from the JSON file
# SERVICE_ACCOUNT_FILE = "weapon-detection-451413-26c91c7aecdb.json"
# credentials = service_account.Credentials.from_service_account_file(
#     SERVICE_ACCOUNT_FILE,
#     scopes=["https://www.googleapis.com/auth/drive.readonly"]
# )

# # Option 1: Upload video from device
# uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# # Option 2: Load video from Google Drive
# drive_link = st.text_input("Or enter Google Drive file ID:")

# if uploaded_file is not None:
#     # Save uploaded video to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         video_path = tmp_file.name

#     # Detect weapons in the uploaded video
#     detect_weapons_in_video(video_path)

# elif drive_link:
#     # Download video from Google Drive
#     video_path = download_video_from_drive(drive_link, credentials)

#     # Detect weapons in the downloaded video
#     detect_weapons_in_video(video_path)
