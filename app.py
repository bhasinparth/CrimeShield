from tkinter import *
from tkinter import filedialog
from surveillance import start_surveillance
from sendemail import email
import numpy as np
import time, pandas
from PIL import Image
from datetime import datetime
import getpass
import time
import cv2
import csv
import os

window=Tk()

window.title("Crime Shield-Security Systems")

def quit(self):
    self.quit()

def About():
    about=Tk()
    about.title("About Scrutiny")
    message = Label( about, text = """
    Scrutiny is a Security Software that provides two features
    1. Train and Create New Members
	2. Automatic Surveillance
	3. Motion and Face Detection
    """ )
    message.pack( side = TOP)
    about.geometry('500x100')

    about.mainloop()

menu = Menu(window)
window.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)

filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)

helpmenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpmenu)
helpmenu.add_command(label="About...", command=About)

#for surveillance
def sur():
    receivers = ['parthbhasin96@gmail.com','shubham7428825138@gmail.com ']
    start_surveillance('automaticsurveillance@gmail.com',receivers,'project1')

#dialog	
def key_dialog():
    key=Tk()
    key.title("Success")
    message=Label(key, text="Congralutions!! You are Successfully added into the list of trusted Party.")
    message.pack(side=TOP)
	
    Button(key, text='Want to Add More??', command=creater).pack(padx=10, pady=5)
    Button(key, text='Start Scrutiny', command=sur).pack(padx=10, pady=5)
    Button(key,text='Exit', command=lambda key=key:quit(key)).pack(padx=10 , pady= 5)	


    key.mainloop()	

#for dataset creation
def creater():
	cam = cv2.VideoCapture(0)
	detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	Id=input('Enter your id ')
	sampleNum=0
	while(True):
		ret, img = cam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = detector.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
			
			#incrementing sample number 
			sampleNum=sampleNum+1
			#saving the captured face in the dataset folder
			cv2.imwrite("dataSet/User."+Id+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])

			cv2.imshow('frame',img)
		#wait for 100 miliseconds 
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break
		# break if the sample number is morethan 20
		elif sampleNum>20:
			break
	cam.release()
	cv2.destroyAllWindows()
    #training                                                              
	recognizer=cv2.face.LBPHFaceRecognizer_create();
	path='dataset'

	def getImagesWithID(path):
		imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
		faces=[]
		IDs=[]
		for imagePath in imagePaths:
			faceImg=Image.open(imagePath).convert('L');
			faceNp=np.array(faceImg,'uint8')
			ID=int(os.path.split(imagePath)[-1].split('.')[1])
			faces.append(faceNp)
			print (ID)
			IDs.append(ID)
			cv2.imshow("training",faceNp)
			cv2.waitKey(10)
		return np.array(IDs),faces

	Ids,faces=getImagesWithID(path)
	recognizer.train(faces,Ids)
	recognizer.write('recognizerr/trainingData.yml')
	cv2.destroyAllWindows()
	key_dialog()

def detection():
	from bokeh.plotting import figure, show, output_file
	from bokeh.models import HoverTool, ColumnDataSource
	from motion_detector import df
	df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
	df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")


	cds=ColumnDataSource(df)

	p=figure(x_axis_type='datetime',height=100, width=500, responsive=True,title="Motion Graph")
	p.yaxis.minor_tick_line_color=None
	p.ygrid[0].ticker.desired_num_ticks=1

	hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
	p.add_tools(hover)

	q=p.quad(left="Start",right="End",bottom=0,top=1,color="green",source=cds)

	output_file("GraphByMotionDetection.html")
	show(p)

def facedetect():
	from live_face_eye import df
	from bokeh.plotting import figure, show, output_file
	from bokeh.models import HoverTool, ColumnDataSource

	df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
	df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")


	cds=ColumnDataSource(df)

	p=figure(x_axis_type='datetime',height=100, width=500, responsive=True,title="Motion Graph")
	p.yaxis.minor_tick_line_color=None
	p.ygrid[0].ticker.desired_num_ticks=1

	hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
	p.add_tools(hover)

	q=p.quad(left="Start",right="End",bottom=0,top=1,color="green",source=cds)

	output_file("GraphByFaceDetection.html")
	show(p)	
   

	
#for icon image of th main page
canvas_width = 300
canvas_height =300
canvas = Canvas(window,
           width=canvas_width,
           height=canvas_height)
canvas.pack()

img = PhotoImage(file="app_icon.png")
canvas.create_image(0,0, anchor=NW, image=img)

message = Label( window, text = "Select any one of the following services" )
message.pack( side = TOP )

b1=Button(window, text="Activate Scrutiny", command= sur)
b1.pack(padx=20, pady=5)
#b1.grid(row=8, column=10, rowspan=8)

b2=Button(window, text="Add a Trusted User For Scrutiny", command= creater)
b2.pack(padx=20, pady=5)
#b2.grid(row=18, column=10, rowspan=10)

b3=Button(window, text="Motion Detection", command= detection)
b3.pack(padx=20, pady=5)
#b2.grid(row=18, column=10, rowspan=10)

b4=Button(window, text="Face Detection", command= facedetect)
b4.pack(padx=20, pady=5)
#b2.grid(row=18, column=10, rowspan=10)
window.mainloop()

