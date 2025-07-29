import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model for mask detection
model = load_model("face_mask_model_final.h5")  # Ensure you have your model saved as "mask_model.h5"

# Define labels
labels = ["With Mask", "Without Mask"]

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Real-Time Photo Capture and Mask Detection
def detect_mask_from_image():
    cap = cv2.VideoCapture(0)  # Access webcam
    print("Press 'Enter' to capture the photo.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display live feed
        cv2.imshow("Press 'Enter' to Capture", frame)
        
        # Wait for 'Enter' key press to capture the photo
        if cv2.waitKey(1) & 0xFF == 13:  # 13 is the ASCII code for the Enter key
            print("Photo captured!")
            process_and_detect_mask(frame)
            break  # Exit the loop after photo is captured

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Function to process the captured photo and detect mask
def process_and_detect_mask(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize list for faces and corresponding predictions
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for model input
        face_resized = cv2.resize(face, (224, 224)) / 255.0  # Normalize
        face_array = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        
        # Predict using the model
        prediction = model.predict(face_array)[0]
        label_idx = np.argmax(prediction)
        label = labels[label_idx]
        confidence = prediction[label_idx]
        
        # Define box color: Green for mask, Red for no mask
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the captured photo with mask prediction
    cv2.imshow("Captured Photo - Face Mask Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_mask_from_image()
