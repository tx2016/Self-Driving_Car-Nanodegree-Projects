import cv2
import glob
import time
import pickle
import numpy as np
from feat_extract import *
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



cars_img = glob.glob("vehicles/*/*.png")
notcars_img = glob.glob("non-vehicles/*/*.png")

# cars = []
# notcars = []

# for car in cars_img:
#     cars.append(cv2.imread(car))
# for ncar in notcars_img:
#     notcars.append(cv2.imread(ncar))

# shuffle images
cars = shuffle(cars)
notcars = shuffle(notcars)


sample_size = 6000
cars = cars_img[0:sample_size]
notcars = notcars_img[0:sample_size]
print('Final size of cars: ', len(cars))
print('Final size of non-cars: ', len(notcars))

# Parameters
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell 
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off




car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=23)

print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
clf = LinearSVC()


# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))


para_file = "para_file.p"

dist_pickle = {}
dist_pickle["scaler"]= X_scaler
dist_pickle["clf"]= clf
dist_pickle["color_space"]= color_space
dist_pickle["orient"]= orient
dist_pickle["pix_per_cell"]= pix_per_cell
dist_pickle["cell_per_block"]=cell_per_block
dist_pickle["hog_channel"]= hog_channel
dist_pickle["spatial_size"]=spatial_size
dist_pickle["hist_bins"]= hist_bins
dist_pickle["spatial_feat"]= spatial_feat
dist_pickle["hist_feat"]= hist_feat
dist_pickle["hog_feat"]= hog_feat

with open(para_file, 'wb') as f:
	pickle.dump(dist_pickle, f)