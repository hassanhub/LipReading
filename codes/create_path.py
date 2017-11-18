import os
import fnmatch
import cv2
import numpy as np
import sys

## This code gets index of the speakers, e.g. python create_path.py 1 2 4 29
if(len(sys.argv)<2):
	print('Insufficient arguments')
	quit()

names=[]
for i in range(1,len(sys.argv)):
	names.append(sys.argv[i])

path='/home/user/LipReading/GRID'

THRESH_FRAME_COUNT=76

file = open(path+'/valid_videos.txt','w')   
file.close()

file = open(path+'/valid_aud_specs.txt','w')   
file.close()

for i in names:
	
	if i==21:
		continue

	print(i)



	os.chdir(path+'/Video/s'+str(i)+'/face')
	numfiles=len(fnmatch.filter(os.listdir(path+'/Video/s'+str(i)+'/face/'), '*.avi'))
	
	for j in range(1,numfiles+1):

		format_num1="{number:06}".format(number=j)
		cap = cv2.VideoCapture(str(format_num1)+'.avi')

		frameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))+1

		if frameCount==1:
			numfiles+=1
			continue
	
		if not frameCount == THRESH_FRAME_COUNT:
			continue

		with open(path+'/valid_videos.txt', 'a') as file1:
			file1.write(path+'/Video/s'+str(i)+'/face/'+str(format_num1)+'.avi\n')
		with open(path+'/valid_aud_specs.txt', 'a') as file2:
			file2.write(path+'/Audio/s'+str(i)+'/AudSpecs/'+str(format_num1)+'.mat\n')


			#
