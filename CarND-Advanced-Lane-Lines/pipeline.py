import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import collections
from thresh import *


global avg_left_fit, avg_right_fit, prev_left_fit, prev_right_fit
num_frames=15
avg_left_fit = collections.deque(maxlen = num_frames)
avg_right_fit = collections.deque(maxlen = num_frames)
prev_left_fit = None
prev_right_fit = None


# Perspective transform on unistorted image
def perspective_transform(img):
    img_size = img.shape[0:2]
    height = img_size[0] #720
    width = img_size[1] #1280
    top_left_x = width * 0.45 #576
    top_right_x = width * 0.55 #704
    top_y = height * 0.63 #453
    dst_top_left_x = width * 0.25 # 320
    dst_top_right_x = width * 0.75 # 960
    
    src = np.float32([[top_left_x, top_y], [top_right_x, top_y],
                       [width, height], [0, height]])
    dst = np.float32([[dst_top_left_x, 0], [dst_top_right_x, 0], 
                      [dst_top_right_x, height], [dst_top_left_x, height]])

    M = cv2.getPerspectiveTransform(src, dst)r
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = i (img, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped, Minv


def detect_lines(binary_warped, left_fit=None, right_fit=None):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Search for initial left and right lanes
    if left_fit is None and right_fit is None:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#         print("leftx_base",leftx_base, "rightx_base",rightx_base)


        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            
#             print('left pixel', good_left_inds)
#             print('len of left pixel', len(good_left_inds))
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    # Use previous left and right lines to detect furthrer lines
    else:
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if (len(leftx) == 0 | len(lefty) == 0):
        left_fit = prev_left_fit
    else:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        
    if (len(rightx) == 0 | len(righty) == 0):
        right_fit = prev_right_fit
    else:
        # Fit a second order polynomial to each
        right_fit = np.polyfit(righty, rightx, 2)

    # Create an output image to draw on and  visualize the result
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, out_img


# Source: http://stackoverflow.com/questions/4151320/efficient-circular-buffer
def smooth_frames(left_fit, right_fit):        
    if (len(avg_left_fit)!=0):
        new_avg_left_fit = np.sum(avg_left_fit, axis=0)/len(avg_left_fit)
        # print('length not zero')
        new_avg_left_fit = 0.95 * new_avg_left_fit + 0.05 * left_fit
    else:
        new_avg_left_fit = left_fit
        # print('length is zero')
    
    if (len(avg_right_fit)!=0):    
        new_avg_right_fit = np.sum(avg_right_fit, axis=0)/len(avg_right_fit)
        new_avg_right_fit = 0.95 * new_avg_right_fit + 0.05 * right_fit
    else:
        new_avg_right_fit = right_fit

    avg_left_fit.append(left_fit)
    avg_right_fit.append(right_fit) 
    
    prev_left_fit = new_avg_left_fit
    prev_right_fit = new_avg_right_fit
    
    return new_avg_left_fit, new_avg_right_fit

##############################################################################
# # Visualize
# # Generate x and y values for plotting
# image = mpimg.imread('test_images/test6.jpg')
# img_undist = undistort(image)
# img_binary = thresh_pipeline(img_undist)
# top_down, Minv = perspective_transform(img_binary)
# histogram = np.sum(top_down[top_down.shape[0]/2:,:], axis=0);
# binary_warped = top_down

# left_fit, right_fit,out_img = detect_lines(binary_warped)

# ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
# # Draw the lane onto the warped blank image

# cv2.polylines(color_warp, np.int_([pts_left]), False, (83, 173, 252), 20)
# cv2.polylines(color_warp, np.int_([pts_right]), False, (83, 173, 252), 20)

# cv2.imwrite('images/top_window.jpg', out_img)
##############################################################################

def calculate_curvature(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_curverad = 0.5*(left_curverad + right_curverad)    
    return avg_curverad

def offset_center(left_fitx, right_fitx):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_x = left_fitx[-1]
    right_x = right_fitx[-1]
    center_x = 0.5*(left_x + right_x)
    offset_x = (1280/2 - center_x) * xm_per_pix
    return offset_x


def add_line_mask(warped, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int_([pts_left]), False, (83, 173, 252), 20)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (83, 173, 252), 20)
    return color_warp


def pipeline(image, prev_left_fit, prev_right_fit):
    img_undist = undistort(image)
    img_binary = thresh_pipeline(img_undist)
    top_down, Minv = perspective_transform(img_binary)
    
    # Current fit coeff and smoothed fit coeff
    #********************************************************************************************#
    left_fit, right_fit, test_img = detect_lines(top_down)
    smooth_left_fit, smooth_right_fit = smooth_frames(left_fit, right_fit)
    #********************************************************************************************#
    # print('left_fit is', left_fit)
    # print('smooth left_fit is ', smooth_left_fit)


    ploty = np.linspace(0, 719, 720)

    # Current fit lines and smoothed fit lines
    #********************************************************************************************#
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    smooth_left_fitx = smooth_left_fit[0]*ploty**2 + smooth_left_fit[1]*ploty + smooth_left_fit[2]
    smooth_right_fitx = smooth_right_fit[0]*ploty**2 + smooth_right_fit[1]*ploty + smooth_right_fit[2]
    #********************************************************************************************#

    curverature = calculate_curvature(ploty, smooth_left_fitx, smooth_right_fitx)
    off_center = offset_center(smooth_left_fitx, smooth_right_fitx)

    color_warp = add_line_mask(img_binary, ploty, smooth_left_fitx, smooth_right_fitx)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(img_undist, 1, newwarp, 0.6, 0)
    cv2.putText(result, "Lane curverature: %.2fm" % curverature, (350, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (235, 235, 50),3)
    cv2.putText(result, "Offset from center: %.2fm" % (off_center), (350, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (235, 235, 50),3)


    warp_zero = np.zeros_like(top_down).astype(np.uint8)
    top_line_fitting = np.dstack((warp_zero, warp_zero, warp_zero))

    #********************************************************************************************#
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    smooth_pts_left = np.array([np.transpose(np.vstack([smooth_left_fitx, ploty]))])
    smooth_pts_right = np.array([np.flipud(np.transpose(np.vstack([smooth_right_fitx, ploty])))])
    #********************************************************************************************#

    # Draw the lane onto the warped blank image
    #********************************************************************************************#
    cv2.polylines(top_line_fitting, np.int_([pts_left]), False, (83, 173, 252), 20)
    cv2.polylines(top_line_fitting, np.int_([pts_right]), False, (83, 173, 252), 20)


    cv2.polylines(top_line_fitting, np.int_([smooth_pts_left]), False, (255, 173, 252), 20)
    cv2.polylines(top_line_fitting, np.int_([smooth_pts_right]), False, (255, 173, 252), 20)
    #********************************************************************************************#

    img_out=np.zeros((720,1920,3), dtype=np.uint8)
    # Original 
    img_out[0:720, 0:1280] = result
    # Threshold
    img_out[0:240,1280:1920] = cv2.resize(np.stack((255*img_binary, 255*img_binary,255*img_binary), axis=2),(640,240))
    # Birds eye view with current and smooth lie fitting
    img_out[240:480,1280:1920] = cv2.resize(top_line_fitting,(640,240))
    # Birds eye view with window sliding box
    img_out[480:720,1280:1920] = cv2.resize(test_img,(640,240))
    
    return img_out, smooth_left_fit, smooth_right_fit

##############################################################################
# Pipeline test
# image = mpimg.imread('test_images/test4.jpg')
# img_undist = undistort(image)
# img_binary = thresh_pipeline(img_undist)
# top_down, Minv = perspective_transform(img_binary)
# out_img,prev_left_fit, prev_right_fit = pipeline(image, prev_left_fit, prev_right_fit)
# cv2.imwrite('images/pipeline_test.jpg', out_img)
##############################################################################


