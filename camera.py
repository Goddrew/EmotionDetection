import cv2
import torch
import numpy as np
from Bless import Bless1
from Bless import labels_map

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoCamera(): 
  def __init__(self): 
    self.video = cv2.VideoCapture(0)
    self.model = Bless1()
    self.model.load_state_dict(torch.load("Bless1.pth", map_location='cpu'))
    self.model.eval()
  
  def __del__(self): 
    self.video.release()
  
  def get_frame(self): 
    ret, frame = self.video.read()
    if not ret: 
      return 
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion = "None"
    for (x, y, w, h) in face_rects: 
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
      roi_gray = gray[y:y+h, x:x+w]
      face = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), 0), 0)
      face = np.concatenate([face, face, face], 1)
      face = torch.tensor(face)
      pred = self.model(face.float())
      emotion = labels_map[f"{torch.argmax(torch.squeeze(pred))}"]
      cv2.putText(frame, emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      break 
    
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

