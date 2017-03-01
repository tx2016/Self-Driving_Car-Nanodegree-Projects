import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pipeline import *
from moviepy.editor import VideoFileClip

prev_left_fit = None
prev_right_fit = None

num_frames=15
avg_left_fit = collections.deque(maxlen = num_frames)
avg_right_fit = collections.deque(maxlen = num_frames)

def process_image(image):
	global prev_left_fit, prev_right_fit
	out_img, prev_left_fit, prev_right_fit = pipeline(image, prev_left_fit, prev_right_fit)
	return out_img

video_input = 'project_video'
# video_input = 'challenge_video'
# video_input = 'harder_challenge_video'
video_output = '{}_solution.mp4'.format(video_input)
clip1 = VideoFileClip('{}.mp4'.format(video_input)).subclip(0, 10)
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(video_output, audio=False)

