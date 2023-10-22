import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0) #default web cam


#loading and encoding the sample user images

bo_image = face_recognition.load_image_file("C:/Users/ssaks/Attendance/Barack Obama.jpeg")
bo_encoding = face_recognition.face_encodings(bo_image)[0]

sg_image = face_recognition.load_image_file("C:/Users/ssaks/Attendance/Selena Gomez.jpg")
sg_encoding = face_recognition.face_encodings(sg_image)[0]

em_image = face_recognition.load_image_file("C:/Users/ssaks/Attendance/Emma Watson.png")
em_encoding = face_recognition.face_encodings(em_image)[0]

ms_image = face_recognition.load_image_file("C:/Users/ssaks/Attendance/M.S.Dhoni.jpeg")
ms_encoding = face_recognition.face_encodings(ms_image)[0]

known_face_encoding = [
    bo_encoding,
    sg_encoding,
    em_encoding,
    ms_encoding
]

known_faces_names=[
    "Barack Obama",
    "Selena Gomez",
    "Emma Watson",
    "M.S. Dhoni"
]

students = known_faces_names.copy() 

face_location = []
face_encodings= []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f=open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read() # first value is signal, second is the actual video input
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
   # as cv2 takes input in bgr format
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance) #to get the best fit
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
            
            #   Entering the details to a csv file
            face_names .append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)   #   To avoid multiple entries
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendence System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the video_capture, cv2 and the file

video_capture.release()
cv2.destroyAllWindows()
f.close()
