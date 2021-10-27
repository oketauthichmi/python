import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np
import sqlite3

window=tk.Tk()
window.title("nhận diện khuôn mặt")


l1=tk.Label(window, text="Mã",font=("Arisan",14))
l1.grid(column=0, row=0)
t1=tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2=tk.Label(window, text="Tên",font=("Arisan",14))
l2.grid(column=0, row=1)
t2=tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)






def train():
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    path='dataSet'
    def getImage(path):
        imagePaths=[os.path.join(path, f)for f in os.listdir(path)]
        faces=[]
        ids=[]
        for imagePath in imagePaths:
            faceImage=Image.open(imagePath).convert('L')
            faceNp=np.array(faceImage,'uint8')
            #print(imagePath)
            Id=int(imagePath.split('\\')[1].split('.')[1])
            #print(Id)
            faces.append(faceNp)
            ids.append(Id)
            cv2.imshow('trainning',faceNp)
            cv2.waitKey(10)
            
        return faces,ids
    
    faces, ids=getImage(path)
    recognizer.train(faces, np.array(ids))
    if not os.path.exists('recognizer'):
        os.makedirs('recognizer')
    recognizer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()
    messagebox.showinfo("result", "complete")
b1=tk.Button(window, text="Tranning", font=("Arisan", 14), fg="blue", command=train)
b1.grid(column=0, row=2)

def face():
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('recognizer/trainningData.yml')
    
    def getprofile(id):
        conn=sqlite3.connect("E:/bc/data.db")
        query="SELECT * FROM people WHERE Id="+str(id)
        cursor=conn.execute(query)
        profile=None
        for row in cursor:
            profile=row
        conn.close()
        return profile
    
    cap=cv2.VideoCapture(0)
    fontface=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            id,confidence=recognizer.predict(roi_gray)
            if confidence<40:
                profile=getprofile(id)
                if(profile!=None):
                    cv2.putText(frame,""+str(profile[1]),(x+10, y+h+30),fontface,1,(0,255,0),2)
            else:
                cv2.putText(frame,"Unknown",(x+10, y+h+30),fontface,1,(0,0,255),2)
        cv2.imshow('show',frame)
        if(cv2.waitKey(1)==ord('q')):
            break
        
    cap.release()
    cv2.destroyAllWindows()
b1=tk.Button(window, text="khuôn mặt", font=("Arisan", 14), fg="green", command=face)
b1.grid(column=1, row=2)

def dataset():
    def insertOrUpdate(id, name):
        conn=sqlite3.connect("E:/bc/data.db")
        query="SELECT * FROM people WHERE Id="+str(id)
        cusror=conn.execute(query)
        isRecordExist=0
        for row in cusror:
            isRecordExist=1
        if(isRecordExist==0):
            query="INSERT INTO people(Id,name) VALUES("+str(id)+",'"+str(name)+"')"
        else:
            query="UPDATE people SET name='"+str(name)+"'WHERE Id="+str(id)
        
        conn.execute(query)
        conn.commit()
        conn.close()
    
    #load thư viện
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    cap=cv2.VideoCapture(0)
    #thêm vào data
    if t1.get()=="" or t2.get()=="":
        messagebox.showinfo("lỗi", "chưa nhập dữ liệu")
    id=t1.get()
    name=t2.get()
    insertOrUpdate(id, name)
    sampleNum=0
    while True:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)
            if not os.path.exists('dataSet'):
                os.makedirs('dataSet')
            sampleNum += 1
            
            cv2.imwrite('dataSet/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h, x:x+w])
            
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        if sampleNum>100:
            break
        
    cap.release()
    cv2.destroyAllWindows()
b1=tk.Button(window, text="dữ liệu", font=("Arisan", 14), fg="red", command=dataset)
b1.grid(column=2, row=2)

window.geometry("800x200")
window.mainloop()