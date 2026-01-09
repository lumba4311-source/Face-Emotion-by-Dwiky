"""
Real-time Face Emotion Detection menggunakan OpenCV
Compatible dengan model SIMPLE
"""

import cv2
import numpy as np
from tensorflow import keras
import time
import os

# Label emosi
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Warna untuk setiap emosi (BGR format)
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 255, 0),    # Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 165, 0), # Orange
    'Neutral': (128, 128, 128) # Gray
}

class EmotionDetector:
    def __init__(self, model_path='best_model.keras'):
        """
        Initialize emotion detector
        """
        print("=" * 70)
        print("🤖 Emotion Detector Initializing...")
        print("=" * 70)
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file '{model_path}' tidak ditemukan!")
            print("Jalankan dulu: python 2_build_model_SIMPLE.py")
            exit(1)
        
        print(f"📂 Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
        
        # Load Haar Cascade untuk face detection
        print("📂 Loading face detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("❌ ERROR: Face detector failed to load!")
            exit(1)
        
        print("✅ Face detector loaded!")
        
        self.emotion_history = []
        self.history_size = 5  # Smoothing dengan 5 frame terakhir
        
        print("=" * 70)
        print("✅ Initialization complete!")
        print("=" * 70)
    
    def preprocess_face(self, face_img):
        """
        Preprocess wajah untuk input ke model
        """
        # Resize ke 48x48
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert ke grayscale jika belum
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        face_img = face_img.astype('float32') / 255.0
        
        # Reshape untuk model
        face_img = face_img.reshape(1, 48, 48, 1)
        
        return face_img
    
    def predict_emotion(self, face_img):
        """
        Predict emosi dari gambar wajah
        """
        try:
            processed_face = self.preprocess_face(face_img)
            predictions = self.model.predict(processed_face, verbose=0)[0]
            
            # Smoothing dengan history
            self.emotion_history.append(predictions)
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            
            # Average predictions
            avg_predictions = np.mean(self.emotion_history, axis=0)
            emotion_idx = np.argmax(avg_predictions)
            confidence = avg_predictions[emotion_idx]
            
            return emotion_labels[emotion_idx], confidence, avg_predictions
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Unknown", 0.0, np.zeros(7)
    
    def draw_emotion_bar(self, frame, predictions, x, y, w):
        """
        Gambar bar chart untuk probabilitas setiap emosi
        """
        bar_height = 20
        bar_width = w
        start_y = max(10, y - len(emotion_labels) * bar_height - 10)
        
        for i, (label, prob) in enumerate(zip(emotion_labels, predictions)):
            bar_y = start_y + i * bar_height
            bar_length = int(prob * bar_width)
            
            # Background bar
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height - 2),
                         (50, 50, 50), -1)
            
            # Probability bar
            color = emotion_colors[label]
            cv2.rectangle(frame, (x, bar_y), (x + bar_length, bar_y + bar_height - 2),
                         color, -1)
            
            # Label dan persentase
            text = f"{label}: {prob*100:.1f}%"
            cv2.putText(frame, text, (x + 5, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def detect_from_webcam(self, camera_index=0, show_probabilities=True):
        """
        Deteksi emosi real-time dari webcam
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open webcam!")
            print("💡 Try:")
            print("   1. Check if webcam is connected")
            print("   2. Change camera_index (try 1 or 2)")
            print("   3. Check webcam permissions")
            return
        
        print("\n" + "=" * 70)
        print("📹 Starting webcam emotion detection...")
        print("=" * 70)
        print("⌨️  Controls:")
        print("   'q' - Quit")
        print("   's' - Toggle probability display")
        print("=" * 70)
        
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Cannot read frame")
                    break
                
                # Flip frame untuk mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(48, 48)
                )
                
                # Process setiap wajah yang terdeteksi
                for (x, y, w, h) in faces:
                    # Extract region wajah
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence, predictions = self.predict_emotion(face_roi)
                    color = emotion_colors[emotion]
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw emotion label
                    label_text = f"{emotion}: {confidence*100:.1f}%"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    
                    # Background untuk text
                    cv2.rectangle(frame,
                                (x, y - label_size[1] - 15),
                                (x + label_size[0] + 10, y),
                                color, -1)
                    
                    # Text
                    cv2.putText(frame, label_text, (x + 5, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Draw probability bars
                    if show_probabilities and y > 150:
                        self.draw_emotion_bar(frame, predictions, x, y, w)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.time()
                    fps = 10 / (current_time - fps_time)
                    fps_time = current_time
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display face count
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Emotion Detection - Press Q to quit', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    show_probabilities = not show_probabilities
                    print(f"Probability display: {'ON' if show_probabilities else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Webcam closed")
    
    def detect_from_image(self, image_path, output_path='result.jpg'):
        """
        Deteksi emosi dari gambar
        """
        if not os.path.exists(image_path):
            print(f"❌ ERROR: Image file '{image_path}' tidak ditemukan!")
            return
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ ERROR: Cannot read image from {image_path}")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        print(f"\n✅ Found {len(faces)} face(s)")
        
        # Process setiap wajah
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = gray[y:y+h, x:x+w]
            emotion, confidence, predictions = self.predict_emotion(face_roi)
            color = emotion_colors[emotion]
            
            # Draw rectangle dan label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            label_text = f"{emotion}: {confidence*100:.1f}%"
            cv2.putText(frame, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            print(f"   Face {i+1}: {emotion} ({confidence*100:.1f}%)")
        
        # Save result
        cv2.imwrite(output_path, frame)
        print(f"\n✅ Result saved to {output_path}")
        
        # Display
        cv2.imshow('Result - Press any key to close', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """
    Main function
    """
    print("\n" + "=" * 70)
    print("😊 FACE EMOTION DETECTION")
    print("=" * 70)
    
    # Initialize detector
    try:
        detector = EmotionDetector('best_model.keras')
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print("\n📋 Select mode:")
    print("   1. Webcam (real-time)")
    print("   2. Image file")
    choice = input("\nChoose option (1/2): ").strip()
    
    if choice == '1':
        print("\n💡 Tip: Ensure good lighting for better detection!")
        detector.detect_from_webcam()
    elif choice == '2':
        image_path = input("\nEnter image path: ").strip()
        # Remove quotes if user included them
        image_path = image_path.strip('"').strip("'")
        detector.detect_from_image(image_path)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        