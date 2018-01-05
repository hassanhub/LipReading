import os
import fnmatch
import cv2
import numpy as np
import sys

## This code gets range of the speakers in dataset, e.g. python prepare_crop_files.py 3 6
if(len(sys.argv)<3):
	print('Insufficient arguments')
	quit()

start=int(sys.argv[1])
end=int(sys.argv[2])


path='/home/user/LipReading/GRID'
os.system('mkdir '+path)
os.system('mkdir '+path+'/Video')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ds_factor = 0.5

for i in range(start,end):

	if i==21:
		continue

	#print(path+'/Video')
	os.chdir(path+'/Video')

	file='s'+str(i)+'.mpg_vcd.zip'
	link='http://spandh.dcs.shef.ac.uk/gridcorpus/s'+str(i)+'/video/'+file

	#downloading and unzipping video for person 1
	#os.system('wget '+link)
	#os.system('unzip '+file)
	#os.system('rm -f -r '+file)

	#renaming 
	#print(path+'/Video/s'+str(i)) 
	os.chdir(path+'/Video/s'+str(i))
	#os.system('ls *.mpg | cat -n | while read n f; do mv "$f" "$(printf %06d $n).mpg"; done')

	#cropping faces, creating new video, stabilizing new video 
	source_path=path+'/Video/s'+str(i)+'/'
	os.mkdir(source_path+'face') 
	os.chdir(source_path+'face')

	numfiles=len(fnmatch.filter(os.listdir(path+'/Video/s'+str(i)), '*.mpg'))
	for j in range(1,numfiles+1):

		format_num1="{number:06}".format(number=j)

		#os.system('mkdir '+source_path+'frames/'+str(format_num1))
		#print('Reading video from : '+source_path+str(format_num1)+'.mpg')

		cap = cv2.VideoCapture(source_path+str(format_num1)+'.mpg')

		print('Writing video : '+source_path+'face/'+str(format_num1)+'.avi')
		out = cv2.VideoWriter()
		fourcc=cv2.cv.CV_FOURCC('m','p','4','v')
		success = out.open(source_path+'face/'+str(format_num1)+'.avi', fourcc, 25.0, (128,128),False)
		print('Success: '+str(success))

		#print('Writing frames to : '+source_path+'frames/'+str(format_num1)+'/')
		count=0

		while(cap.isOpened()):
			count=count+1
			ret, frame = cap.read()
			if ret==False:
				break
			#cropping face
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			if count==1:
				face_rects = face_cascade.detectMultiScale(gray,1.05,3,minSize=(128,128))
			if face_rects==():
        		    #count-=1
        		    #continue;
        		    break

			x,y,w,h= face_rects[0]
			#136,210,49,29
			#inc=30
			#x=136-inc
			#y=210-int(inc/2)
			#w=49+(2*inc)
			#h=29+(inc)
			roi=gray[y:y+h,x:x+w]
			#print(str(x)+','+str(y)+','+str(w)+','+str(h))
			#resizing 
			roi=cv2.resize(roi,(128,128))

			#cv2.imwrite(source_path+'frames/'+str(format_num1)+'/'+str(format_num2)+'.jpg',roi,)
			#writing video (unstabilized)
			out.write(roi)
		cap.release()
		out.release()

		#with open(path+'/log.txt', 'a') as file:
		#	file.write(source_path+'frames/'+str(format_num1)+'.mpg '+str(count)+' '+str(success))

		#stabilizing video 
		#os.system('ffmpeg -i '+ str(format_num1)+'.mpg' +' -vf vidstabdetect -f null -')
		#os.system('ffmpeg -i '+ str(format_num1)+'.mpg' +' -vf vidstabtransform=smoothing=5:input="transforms.trf" '+ 's_'+str(format_num1)+'.mpg' )

for i in range(start,end):
	
	if i==21:
		continue

	print(i)

	os.system('mkdir '+path+'/Audio/s'+str(i))

	os.chdir(path+'/Video/s'+str(i))
	numfiles=len(fnmatch.filter(os.listdir(path+'/Video/s'+str(i)), '*.mpg'))
	for j in range(1,numfiles+1):

		format_num1="{number:06}".format(number=j)
		#os.system('ffmpeg -i '+ str(format_num1)+'.mpg' +' -q:a 0 -map a '+path+'/Audio/s'+str(i)+'/'+str(format_num1)+'.wav' )
		os.system('ffmpeg -i '+ str(format_num1)+'.mpg' +' -ac 1 -ar 8000 '+path+'/Audio/s'+str(i)+'/'+str(format_num1)+'.wav' )

