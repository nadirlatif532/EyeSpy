# coding: utf-8
# from data_provider import *
from C3D_model import *
import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os 
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import skimage.io as io
from skimage.transform import resize
import h5py
import cv2
import shutil
from PIL import Image
import sys
import Main
import anomalydetector
from kivy.utils import rgba
from kivy.uix.label import Label
import GPUtil

def feature_extractor(OUTPUT_DIR_TEXT,VIDEO_PATH,TEMP_PATH, EXTRACTED_LAYER = 6,RUN_GPU = True, BATCH_SIZE = 8 ):
	"""

	:param OUTPUT_DIR_TEXT:
	:param VIDEO_PATH:
	:param TEMP_PATH:
	:param EXTRACTED_LAYER:
	:param RUN_GPU:
	:param BATCH_SIZE:
	:return:
	"""
	GPUs = GPUtil.getGPUs()
	Gmem = GPUs[0].memoryTotal
	if(Gmem < 6144 and Gmem > 4096):
		BATCH_SIZE=4
	elif(Gmem <= 4096):
		BATCH_SIZE=2
	else:
		BATCH_SIZE=BATCH_SIZE

	resize_w = 112
	resize_h = 171
	nb_frames = 16
	net = C3D(487)
	print('net', net)
	net.load_state_dict(torch.load('./c3d.pickle'))
	if RUN_GPU : 
		net.cuda(0)
		net.eval()

	feature_dim = 4096 if EXTRACTED_LAYER != 5 else 8192
	mainmenu = Main.App.get_running_app().root.get_screen("MainMenu")
	mainmenu.ids.videoplayer.state = 'stop'
	if mainmenu.Batch_Flag == False:
		mainmenu.popup.content = Label(text='Features are being extracted..(1/3)', color=rgba('#DAA520'), font_size=24)
	else:
		mainmenu.popup.open()
	gpu_id = 0


	# current location
	temp_path = TEMP_PATH


	error_fid = open('error.txt', 'w')
	video_path = VIDEO_PATH
	video_name = os.path.basename(video_path)
	print('video_name', video_name)
	print('video_path', video_path)
	frame_path = os.path.join(temp_path, 'frames')
	if not os.path.exists(frame_path):
		os.mkdir(frame_path)


	print('Extracting video frames ...')
	# using ffmpeg to extract video frames into a temporary folder
	# example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg

	cap = cv2.VideoCapture(video_path)
	count = 1
	while (cap.isOpened()):
		ret, frame =cap.read()
		if (ret!= True):
			break

		cv2.imwrite(os.path.join(frame_path,'image_{}.jpg').format(count),frame)
		count += 1



	print('Extracting features ...')
	total_frames = len(os.listdir(frame_path))
	if total_frames == 0:
		error_fid.write(video_name+'\n')
		print('Fail to extract frames for video: %s'%video_name)


	valid_frames = total_frames / nb_frames * nb_frames
	n_feat = valid_frames / nb_frames
	n_batch = n_feat / BATCH_SIZE
	if n_feat - n_batch*BATCH_SIZE > 0:
		n_batch = n_batch + 1
	print('n_frames: %d; n_feat: %d; n_batch: %d'%(total_frames, n_feat, n_batch))


	features = []

	for i in range((int)(n_batch)-1):
		input_blobs = []
		for j in range(BATCH_SIZE):
			clip = []
			clip = np.array([resize(io.imread(os.path.join(frame_path, 'image_{:01d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range((i*BATCH_SIZE+j) * nb_frames+1, min((i*BATCH_SIZE+j+1) * nb_frames+1, valid_frames+1))])
			clip = clip[:, 8: 112, 30: 142, :]
			input_blobs.append(clip)
		input_blobs = np.array(input_blobs, dtype='float32')
		print('Extracting Features..(' + str(i) + '/' + str(int(n_batch - 1)) + ')')
		input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
		input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
		_, batch_output = net(input_blobs, EXTRACTED_LAYER)
		batch_feature  = (batch_output.data).cpu()
		features.append(batch_feature)

	# The last batch
	input_blobs = []
	for j in range((int)(n_feat-(n_batch-1)*BATCH_SIZE)):
		clip = []
		clip = np.array([resize(io.imread(os.path.join(frame_path, 'image_{:01d}.jpg'.format(k))),
								output_shape=(resize_w, resize_h), preserve_range=True) for k in
						 range(int((((n_batch - 1) * BATCH_SIZE + j) * nb_frames + 1)),
							   min(int(((n_batch - 1) * BATCH_SIZE + j + 1) * nb_frames + 1), valid_frames + 1,
								   int((((n_batch - 1) * BATCH_SIZE + j) * nb_frames + 1) + 15)))])

		clip = clip[:, 8: 112, 30: 142, :]
		input_blobs.append(clip)
	input_blobs = np.array(input_blobs, dtype='float32')
	input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
	input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
	_, batch_output = net(input_blobs, EXTRACTED_LAYER)
	batch_feature  = (batch_output.data).cpu()
	features.append(batch_feature)

	features = torch.cat(features, 0)
	features = features.numpy()
	Segments_Features = np.zeros((32, 4096))


	frameclips = np.size(features, 0)

	clipsegments = np.round(np.linspace(0, frameclips, 32))

	count = 0

	for segment in range(0, clipsegments.size - 1):

		clipstart = clipsegments[segment]
		clipend = clipsegments[segment + 1] - 1
		if segment == clipsegments.size:
			clipend = clipsegments[segment + 1]

		if (clipstart == clipend):

			try:
				temp_vect = features[int(clipstart), :]
			except:
				temp_vect = features[int(clipstart - 1), :]

		elif (clipend < clipstart):
			try:
				temp_vect = features[int(clipstart), :]
			except:
				temp_vect = features[int(clipstart - 1), :]

		else:
			temp_vect = np.mean(features[int(clipstart):int(clipend), :], axis=0)

		temp_vect = (temp_vect / np.linalg.norm(temp_vect))

		if np.linalg.norm(temp_vect) == 0:
			print('??')

		Segments_Features[count, :] = temp_vect
		count = count + 1

	result = np.matrix(Segments_Features)
	with open(OUTPUT_DIR_TEXT + '{}.txt'.format(video_name), 'wb') as f:
		for line in result:
			np.savetxt(f, line, fmt='%.6f')

	# clear temp frame folders
	try:
		shutil.rmtree(frame_path)
	except:
		pass



	sniplist = anomalydetector.anomalydetector(VIDEO_PATH,OUTPUT_DIR_TEXT,video_name)

	if mainmenu.Batch_Flag == False:
		print(sniplist)
		scorelist = (sniplist[:,2])
		scorelist = scorelist.astype(float)
		Severity_High = 'Media/severity_high.png'
		Severity_Medium = 'Media/severity_medium.png'
		Severity_Low = 'Media/severity_low.png'
		DR = Main.DisplayRoot()

		for snip_vids in sniplist:
			severity = float(snip_vids[2])
			thumbnail_path = os.path.join(TEMP_PATH,snip_vids[1])
			print(thumbnail_path)
			if severity >= max(scorelist) :
				DR.add_widget(Main.Snippet(thumbnail_path,Severity_High,snip_vids[0]))
			elif severity > min(scorelist) and severity < max(scorelist):
				DR.add_widget(Main.Snippet(thumbnail_path, Severity_Medium,snip_vids[0]))
			elif severity <= min(scorelist):
				DR.add_widget(Main.Snippet(thumbnail_path, Severity_Low,snip_vids[0]))
			else:
				pass

		mainmenu = Main.App.get_running_app().root.get_screen("MainMenu")

		SS = mainmenu.SS
		SS.add_widget(DR)
		mainmenu.ids.Snippets.add_widget(SS)
		mainmenu.ids.videoplayer.state = 'play'
	else:
		try:
			f = open('./Appdata/config.txt')
			lines = f.readlines()
			f.close()
			video_output_path = lines[1]
			video_output_path = video_output_path[:-1]
			output_path = video_output_path
		except:
			output_path = 'Appdata/output/'
		vidname = os.path.basename(VIDEO_PATH)
		list_path = 'Appdata/temp/snip/'
		Batch_SnippetList = []

		for snip_vids in sniplist:
			Batch_SnippetList.append(snip_vids[0])


		with open('Appdata/temp/snip/snippets.txt', 'w') as file:
			for element in Batch_SnippetList:
				file.writelines('file ' + '\'' + element + "\'\n")
			file.close()
		os.system(
			"ffmpeg -f concat -i {}snippets.txt -codec copy {}/{}_anomalous.mp4".format(list_path, output_path,

																						vidname))
		if os.path.exists('./Appdata/temp'):
			shutil.rmtree("./Appdata/temp/")

		os.makedirs('./Appdata/temp')
		os.makedirs('./Appdata/temp/snip')
		os.makedirs('./Appdata/temp/frames')
		os.makedirs('./Appdata/temp/textfeatures')
		os.makedirs('./Appdata/temp/plot')

