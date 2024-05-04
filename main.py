import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

class FaceRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition App")
        
        self.add_face_button = tk.Button(self.window, text="Add Face", command=self.add_face)
        self.add_face_button.pack(pady=10)
        
        self.recognize_button = tk.Button(self.window, text="Recognize", command=self.recognize_face)
        self.recognize_button.pack(pady=10)
        
        self.window.mainloop()
        
    def add_face(self):
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            if len(faces) > 0:
                name = simpledialog.askstring("Name", "Enter your name:")
                if name:
                    cv2.imwrite(f"known_faces/{name}.jpg", frame)
                    messagebox.showinfo("Success", "Face added successfully.")
                    break
                else:
                    messagebox.showerror("Error", "Name not provided.")
        
        video_capture.release()
        cv2.destroyAllWindows()
        
    def recognize_face(self):
        known_faces = []
        known_names = []
        known_face_dir = 'known_faces/'
        for file in os.listdir(known_face_dir):
            if file.endswith('.jpg'):
                image = cv2.imread(os.path.join(known_face_dir, file))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                known_faces.append(gray)
                known_names.append(file.split('.')[0])
        
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                label = self.recognize(roi_gray, known_faces, known_names)
                messagebox.showinfo("Recognition Result", f"Face recognized as {label}")
        
        video_capture.release()
        cv2.destroyAllWindows()
        
    def recognize(self, face, known_faces, known_names):
        for i, known_face in enumerate(known_faces):
            result = cv2.matchTemplate(face, known_face, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, _) = cv2.minMaxLoc(result)
            if max_val > 0.8:  # Adjust this threshold according to your needs
                return known_names[i]
        return "Unknown"

if __name__ == "__main__":
    app = FaceRecognitionApp()
