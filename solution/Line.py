import pickle
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
from Helper_Functions import *
from ObjectDetection import *


xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Event listener, if button of mouse is clicked
def onclick(event, x, y, flags, param):
    global toggle

    if event == cv2.EVENT_LBUTTONDOWN:    
        #just toggle a global variable
        toggle = True
        print(toggle)
    elif event == cv2.EVENT_RBUTTONDOWN:    
        #just toggle a global variable
        toggle = False
        print(toggle)
    return

adaptive = False

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,orientation):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = [] 
        #y values for detected line pixels
        self.ally = []
        #store the history of polyfit coefficients
        self.coeff_history = []
        #center line position
        self.center_line_point = -1
        # orientation can be left or right lane
        self.orientation = orientation
        # number of how many invalid frames have been found in a row --> reset after 5 invalid frames in a row
        self.number_of_subsequent_invalid = 0
        # number of how many valid frames have been found in a row  
        self.number_of_subsequent_valid = 0
        # define the depth until when old results are kept for average calculation    
        self.historyDepth = 5


    def reset(self):
       # reset!
        self.number_of_subsequent_invalid = 0                
        self.coeff_history = []
        self.center_line_point = -1
        self.allx = [] 
        self.ally = [] 
        self.diffs = np.array([0,0,0], dtype='float') 
        self.radius_of_curvature = None 
        self.current_fit = [np.array([False])]  
        self.bestx = None     
        self.best_fit = None  
        self.radius_of_curvature = None
        self.detected = True    #Required for a restart!!
        # skip values!

    # process the pixels found 
    def processLanePts(self, x_pts, y_pts,img_shape):

 
        # initial assumption is that a line will be detected
        self.detected = True

        if len(y_pts)<500 or len(x_pts)<500:
            self.allx = []
            self.ally = []
            self.number_of_subsequent_valid = 0
            self.number_of_subsequent_invalid += 1
            return


        lane_fit = np.polyfit(y_pts, x_pts, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]


        if len(self.coeff_history)>0:
            mean_coeff = np.mean(self.coeff_history, axis=0)
        else:
            #first run! 
            self.coeff_history.append(lane_fit)
            mean_coeff = np.mean(self.coeff_history, axis=0)
 
        relative_coeff_a_change = abs((lane_fit[0] - mean_coeff[0])/mean_coeff[0])
        relative_coeff_b_change = abs((lane_fit[1] - mean_coeff[1])/mean_coeff[1])
        relative_coeff_c_change = abs((lane_fit[2] - mean_coeff[2])/mean_coeff[2])


        relative_coeff_change_sum = relative_coeff_a_change + relative_coeff_b_change + relative_coeff_c_change

        temp_center_line_point = lane_fit[0]*720**2 + lane_fit[1]*720 + lane_fit[2]


        if (relative_coeff_a_change > 2 or relative_coeff_b_change > 2 or relative_coeff_c_change > 2) and relative_coeff_change_sum > 5:
            if abs(lane_fit[0]) < 0.0001 and abs(lane_fit[1]) < 0.2:
                # we assume rather a straight line!!
                print('Straight line assumed')
            else:
                # Points seem to be invalid
                self.detected = False


        sign_change_a = False
        sign_change_b = False

        if (lane_fit[0] > 0 and self.coeff_history[len(self.coeff_history)-1][0] < 0) or (lane_fit[0] < 0 and self.coeff_history[len(self.coeff_history)-1][0] > 0):
            sign_change_a = True

        if (lane_fit[1] > 0 and self.coeff_history[len(self.coeff_history)-1][1] < 0) or (lane_fit[1] < 0 and self.coeff_history[len(self.coeff_history)-1][1] > 0):
            sign_change_b = True


        if sign_change_a==True and sign_change_b==True:
               # Points seem to be invalid
                self.detected = False
  
        if self.number_of_subsequent_invalid > 5:
            self.reset()
                #print('Frame seems to be invalid!!!')
        else:
            # do nothing
            print()


        if self.detected == False:
            self.number_of_subsequent_invalid = self.number_of_subsequent_invalid + 1
            if (len(self.coeff_history) >= self.historyDepth):
                self.coeff_history.pop(0)
            # step 2: append newly found coeffs
            self.coeff_history.append(lane_fit)
            self.number_of_subsequent_valid = 0  # 

            # step 6: set the mean best fit
            mean_coeff = np.mean(self.coeff_history, axis=0)
            self.best_fit = mean_coeff  
            self.current_fit = self.best_fit          

        else:   
            # valid points found!!
            self.detected = True
            self.number_of_subsequent_valid += 1  # 
            self.number_of_subsequent_invalid = 0  # reset the invalid counter
            #print('Frame seems to be valid!!!')
            # step 1: remove first frame coeffs if more than threshold items available
            if (len(self.coeff_history) >= self.historyDepth):
                self.coeff_history.pop(0)
            # step 2: append newly found coeffs
            self.coeff_history.append(lane_fit)
 
            # step 3: append x/y points
            self.allx = x_pts
            self.ally = y_pts

            # step 4: set current fit polynomial coefficients
            self.current_fit = lane_fit

            # step 5: set current fit polynomial coefficients
            self.diffs = [(lane_fit[0] - mean_coeff[0]) , (lane_fit[1] - mean_coeff[1]), (lane_fit[2] - mean_coeff[2])]

            # step 6: set the mean best fit
            mean_coeff = np.mean(self.coeff_history, axis=0)
            self.best_fit = mean_coeff


            # step 7: set the center_line
            self.center_line_point = temp_center_line_point


            # step 6: set the curvature radius
            self.radius_of_curvature = self.calculateCurvature()

    
            self.current_fit = self.best_fit          
            #step 7: distance in meters of vehicle center from the line
            if self.orientation is 'left':
                self.line_base_pos = self.center_line_point - 300
            else:
                self.line_base_pos = 900 - self.center_line_point                 


        # print('-----------------------------------------------------')
        # print('Process ' , len(y_pts) ,  ' points on Line ' , self.orientation)
        # print("Coeff a " ,lane_fit[0])
        # print("Coeff b" ,lane_fit[1])
        # print("Coeff c" ,lane_fit[2])       
        # print("Mean Coeff a" ,mean_coeff[0])
        # print("Mean Coeff b" ,mean_coeff[1])
        # print("Mean Coeff c" ,mean_coeff[2])
        # print("relative_coeff_a_change " ,relative_coeff_a_change)
        # print("relative_coeff_b_change " ,relative_coeff_b_change)
        # print("relative_coeff_c_change " ,relative_coeff_c_change)
        # print("center_line_point " ,self.center_line_point)
        # print("line_base_pos" ,self.line_base_pos)
        # print('-----------------------------------------------------')


    def calculateCurvature(self):
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = self.current_fit[0] 

        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=300 for left, and x=900 for right)
        if self.orientation == 'left':
            points_x = np.array([300 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                      for y in ploty])
        else:
            points_x = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                        for y in ploty])

        points_x = points_x[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line
        points_x_fit = np.polyfit(ploty, points_x, 2)
        fitx = points_x_fit[0]*ploty**2 + points_x_fit[1]*ploty + points_x_fit[2]

        # Plot up the fake data
        #mark_size = 3
        # plt.plot(points_x, ploty, 'o', color='red', markersize=mark_size)
        # plt.xlim(0, 1280)
        # plt.ylim(0, 720)
        # plt.plot(fitx, ploty, color='green', linewidth=3)
        # plt.gca().invert_yaxis() # to visualize as we do the images
        # plt.show()

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        #curverad = ((1 + (2*points_x_fit[0]*y_eval + points_x_fit[1])**2)**1.5) / np.absolute(2*points_x_fit[0])

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


        #print(left_curverad, right_curverad)        
        return curverad.astype(int)


# Define a class to receive the characteristics of each line detection
class LeftLine(Line):
    def __init__(self):
         Line.__init__(self, "left")


# Define a class to receive the characteristics of each line detection
class RightLine(Line):
    def __init__(self):
         Line.__init__(self, "right")

# The EgoLane contains a leftline and a rightline object
class EgoLane():
    def __init__(self):
        self.leftline = LeftLine()
        self.rightline = RightLine()

    # processing pipeline for a found frame    
    def pipeline(self, frame,displayText=True):
        global adaptive

        #1.Step: take the modified image after contrastIncrease()
        img = frame.modifiedImg

        #2. Step: undistort this image
        img_undistort = frame.camera.undistort(img)

        #3. Step: apply the color gradient
        colorGrad = self.colorGradient(img_undistort,(170,220),(22,100))

        #4. Step: Warp the image
        warped = frame.camera.warp(colorGrad)

        #5.Step: mask the area of interest
        maskedImage = frame.camera.maskAreaOfInterest(warped)

        #6.Step: convert to black/white
        grayImage = frame.camera.rgbConvertToBlackWhite(maskedImage)
        
        #7.Step in case we have nothing detected yet --> detect newly
        #   in case we have already detected lines --> detect from this base
        if self.leftline.detected == True and self.leftline.detected == True:
            histoCurvatureFitImage = self.nextFramehistoCurvatureFit(grayImage)
            #histoCurvatureFitImage = self.histoCurvatureFit(grayImage)
        else:
            histoCurvatureFitImage = self.histoCurvatureFit(grayImage)


        # 8.Step Now display the found lines and plot them on top of the original image
        coloredLaneImage = self.displayLane(frame.currentImg)
       

        #9.Step Now add a small resized image of the curvature calculation in the upper middle of the original image
        grayImage = np.uint8(grayImage)
        gray2color = cv2.cvtColor(grayImage,cv2.COLOR_GRAY2RGB ,3)
        gray2color = cv2.addWeighted(coloredLaneImage.astype(np.float32)*255, 1, (gray2color.astype(np.float32))*255, 1, 0)
        resized_image = cv2.resize(gray2color,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)

        #10.Step Unwarp the whole image
        unwarped = frame.camera.unwarp(coloredLaneImage)*255
        src_mask = mask = 255 * np.ones(resized_image.shape, resized_image.dtype)
        unwarped = cv2.seamlessClone(resized_image.astype(np.uint8), unwarped.astype(np.uint8), src_mask.astype(np.uint8), (640,200), cv2.NORMAL_CLONE)


        #11.Step: add additional text left/right which might be interesting
        if displayText==True:
            new_image = np.zeros_like(unwarped)

            text1 = "Left Lane Dropout Counter: " + str(self.leftline.number_of_subsequent_invalid)
            text1a = "Left Lane Points found: " + str(len(self.leftline.allx))
            text2 = "Right Lane Dropout Counter: " + str(self.rightline.number_of_subsequent_invalid)
            text2a = "Right Lane Points found: " + str(len(self.rightline.allx))
            text3 = "Curvature radius left: " + str(self.leftline.radius_of_curvature) + " (m)"
            text4 = "Curvature radius right: " + str(self.rightline.radius_of_curvature) + " (m)"

            center_deviation = round((self.leftline.line_base_pos*xm_per_pix*100 + self.rightline.line_base_pos*xm_per_pix*(-100))/2,2)

            if center_deviation >=0:
                text5 = "Vehicle is left of center " + str(center_deviation) + ' (cm)'
            else:
                text5 = "Vehicle is right of center " + str(center_deviation) + ' (cm)'

            text6 = text5
            text7 = "RESTART! " 
            text8 = "ADAPTIVE!!"

            cv2.putText(new_image,text1,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text1a,(50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text2,(900,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text2a,(900,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text3,(50,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text4,(900,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text5,(50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(new_image,text6,(900,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1,cv2.LINE_AA)
            
            if (self.leftline.number_of_subsequent_invalid == 5 or self.rightline.number_of_subsequent_invalid == 5):
                cv2.putText(new_image,text7,(540,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)

            if adaptive == True:
                cv2.putText(new_image,text8,(540,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)                

            result = cv2.addWeighted(unwarped.astype(np.float32)*255, 1, (new_image.astype(np.float32))*255, 1, 0)
        else:
            result = unwarped/255

        #12. Detect whether we should change preprocessing of that image in order to get more pixels
        if len(self.leftline.allx) < 500:
            adaptive = True
            self.leftline.reset()
        elif len(self.rightline.allx) < 500:
            adaptive = True
            self.rightline.reset()
        else:
            if self.leftline.number_of_subsequent_valid > 0:
                adaptive = False            

        #13.Step finally return the result
        return result


    def displayLane(self,img):
        overlay_img = np.zeros_like(img)

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

        if self.leftline.detected == True:
            #left_mean_coeff = self.leftline.current_fit
            left_mean_coeff = self.leftline.best_fit
            left_fitx = left_mean_coeff[0]*ploty**2 + left_mean_coeff[1]*ploty + left_mean_coeff[2]

            for x1,y1 in zip(left_fitx.astype(int),ploty.astype(int)):
                cv2.circle(overlay_img,(x1,y1),4,(255,255, 0),2)
        else:
            left_mean_coeff = self.leftline.best_fit
            left_fitx = left_mean_coeff[0]*ploty**2 + left_mean_coeff[1]*ploty + left_mean_coeff[2]
            for x1,y1 in zip(left_fitx.astype(int),ploty.astype(int)):
                cv2.circle(overlay_img,(x1,y1),4,(255, 0, 255),2)

        if self.rightline.detected == True:
            #right_mean_coeff = self.rightline.current_fit
            right_mean_coeff = self.rightline.best_fit
            right_fitx = right_mean_coeff[0]*ploty**2 + right_mean_coeff[1]*ploty + right_mean_coeff[2]
            for x1,y1 in zip(right_fitx.astype(int),ploty.astype(int)):
                cv2.circle(overlay_img,(x1,y1),4,(255,255, 0),2)
        else:
            right_mean_coeff = self.rightline.best_fit
            right_fitx = right_mean_coeff[0]*ploty**2 + right_mean_coeff[1]*ploty + right_mean_coeff[2]
            for x1,y1 in zip(right_fitx.astype(int),ploty.astype(int)):
                cv2.circle(overlay_img,(x1,y1),4,(255, 0, 255),2)


        if self.rightline.detected == True and self.leftline.detected == True:
            for x1,y1,x2,y2 in zip(left_fitx.astype(int),ploty.astype(int),right_fitx.astype(int),ploty.astype(int)):
                cv2.line(overlay_img,(x1,y1),(x2,y2),(0, 255, 0),2)
        elif self.leftline.best_fit != None and self.rightline.best_fit != None:
            for x1,y1,x2,y2 in zip(left_fitx.astype(int),ploty.astype(int),right_fitx.astype(int),ploty.astype(int)):
                cv2.line(overlay_img,(x1,y1),(x2,y2),(0, 255, 0),2)

        return overlay_img/255

    def processFrame(self, frame, displayText=True):
        #print("Processing egolane")
        return self.pipeline(frame,displayText)

    def colorGradient(self, img, s_channel_thresh=(180, 255), sobel_x_thresh=(30, 120)):
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        # Sobel x
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        #abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        #scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Sobel y
        #sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
        #abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from vertical
        #scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

        #mag_sobel = np.sqrt(sobel_x*sobel_x + sobel_y*sobel_y)
        #abs_sobel = np.absolute(mag_sobel)
        abs_sobel = np.absolute(sobel_x)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sobel_x_thresh[0]) & (scaled_sobel <= sobel_x_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_channel_thresh[0]) & (s_channel <= s_channel_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( sxbinary, np.zeros_like(sxbinary), s_binary))
        return color_binary
    
    def histoCurvatureFit(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        height = int(binary_warped.shape[0]/2)
        histogram = np.sum(binary_warped[height:,:], axis=0)

        #print(histogram)
        #plt.plot(histogram)
        #plt.show()

        #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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
        margin = 40
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
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # now process the pixel for left line and for the right line separately
        self.leftline.processLanePts(leftx, lefty, binary_warped.shape)
        self.rightline.processLanePts(rightx, righty, binary_warped.shape)

        ###############################################
        # here only the plotting part starts 
        ###############################################


        left_fit = self.leftline.current_fit
        right_fit = self.rightline.current_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # fill the lane with red
        for x1,y1,x2,y2 in zip(left_fitx.astype(int),ploty.astype(int),right_fitx.astype(int),ploty.astype(int)):
            cv2.line(window_img,(x1,y1),(x2,y2),(255,0, 0),2)
            cv2.circle(window_img,(x1,y1),2,(255,255, 0),2)
            cv2.circle(window_img,(x2,y2),2,(255,255, 0),2)
        
        # Draw the lane onto the warped blank image in green color
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')

        #print(window_img)
        #plt.imshow(result/255)
        #plt.show()
        return result/255
        # plt.imshow(result/255)
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # result = (result/255)

    def nextFramehistoCurvatureFit(self, binary_warped):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        if self.leftline.current_fit != None and self.leftline.current_fit[0] != False and self.rightline.current_fit != None and self.rightline.current_fit[0] != False:
            left_fit = self.leftline.current_fit
            right_fit = self.rightline.current_fit
        elif self.leftline.best_fit != None and self.leftline.best_fit[0] != False and self.rightline.best_fit != None and self.rightline.best_fit[0] != False:
            left_fit = self.leftline.best_fit
            right_fit = self.rightline.best_fit
        else:
            return self.histoCurvatureFit(binary_warped)

        #print('left_fit = ', left_fit) 
        #print('right_fit = ', right_fit) 

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 50
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

       # now process the pixel for left line and for the right line separately
        self.leftline.processLanePts(leftx, lefty, binary_warped.shape)
        self.rightline.processLanePts(rightx, righty, binary_warped.shape)

        ###############################################
        # here only the plotting part starts 
        ###############################################


        left_fit = self.leftline.current_fit
        right_fit = self.rightline.current_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # fill the lane with red
        for x1,y1,x2,y2 in zip(left_fitx.astype(int),ploty.astype(int),right_fitx.astype(int),ploty.astype(int)):
            cv2.line(window_img,(x1,y1),(x2,y2),(255,0, 0),2)
            cv2.circle(window_img,(x1,y1),2,(255,255, 0),2)
            cv2.circle(window_img,(x2,y2),2,(255,255, 0),2)
        
        # Draw the lane onto the warped blank image in green color
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')

        #print(window_img)
        #plt.imshow(result/255)
        #plt.show()
        return result/255

# Define a class to receive the characteristics of each line detection
class Frame():
    def __init__(self):
        self.currentImg = None
        self.modifiedImg = None
        self.currentEgoLaneOverlay = None
        self.egoLane = EgoLane()
        self.objectDetection = ObjectDetection('../svm_cal/svc_pickle.p')
        self.camera = None

    # load an image from a file
    def loadImageFromFile(self, filename):
        self.currentImg = cv2.imread(filename)
        self.currentImg = cv2.cvtColor(self.currentImg, cv2.CV_RGB2BGR) 

    # process the current frame    
    def processCurrentFrame(self,counter):
    	#1.Step: reset the overlay image
        self.currentEgoLaneOverlay = None

        #2.Step: initialize the object detection in order to be able to receive new frames and calculate object tracking
        self.objectDetection.initNextFrame()
        
        #3.Step: convert to RBG (unfortunately the SVM was trained for RBG images.....)
        dummy = self.currentImg[...,::-1]
        #4.Step: calculate the objects
        self.objectDetection.processFrame(dummy ,counter,False)

        #5.Step: process the lane finding and create the overlay image
        self.currentEgoLaneOverlay = self.egoLane.processFrame(self,True)

    # helper method which just displays the current image
    def displayCurrentImage(self, overlay=True):

        if overlay == True and self.currentEgoLaneOverlay != None:
            print("Show Overlay")
            
            img_pipelined = np.uint8(255*self.currentEgoLaneOverlay/np.max(self.currentEgoLaneOverlay))
            result = cv2.addWeighted(self.currentImg.astype(int), 1, img_pipelined.astype(int), 0.5, 0,dtype=cv2.CV_8U)
            
            plt.imshow(result)
        else:
            print("No Overlay!")
            plt.imshow(self.currentImg)

        plt.title('Input Image')
        plt.show()

    # return the overlay image if available
    def getOverlayImage(self):
        if self.currentEgoLaneOverlay != None:
            
            # 1.Step: get the pipelined image
            img_pipelined = np.uint8(255*self.currentEgoLaneOverlay/np.max(self.currentEgoLaneOverlay))

            #2.Step: draw the pipelined image on top of currentImage
            result = cv2.addWeighted(self.currentImg.astype(int), 1, img_pipelined.astype(int), 0.5, 0,dtype=cv2.CV_8U)

            #3.Step: Draw the object boxes
            result = draw_objects(result, self.objectDetection.objectsDetected)
            
            return result
        else:
            return self.currentImg



    # initialize the camera with the calibration data from the presiously created pickle
    def initializeCamera(self, fileName='../camera_cal/camera_calibration_pickle.p'):
        self.camera = Camera(fileName)

    def initializeSVMConfiguration(self, fileName='../svm_cal/svc_pickle.p'):
        self.objectDetection = ObjectDetection(fileName)

# Camera class which contains all relevant camera information
class Camera():
    def __init__(self,fileName):
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None
        self.rightLaneCenter = None       # this is actually the right lower point of the warped image
        self.leftLaneCenter = None        # this is actually the left lower point of the warped image
        self.calibrationFileName = fileName

        if os.path.isfile(fileName) == True:
            self.loadCameraCalibration()
        else:
            print("No Camera calibration found.... Exiting().....")
            #TODO: auto-calibrate
            exit()

    # load the camera calibration 
    def loadCameraCalibration(self):
        # Read in the saved camera matrix and distortion coefficients
        # These are the arrays you calculated using cv2.calibrateCamera()
        dist_pickle = pickle.load( open(self.calibrationFileName , "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        self.M = dist_pickle["M"]
        self.Minv = dist_pickle["Minv"]
        self.rightLaneCenter = dist_pickle["rightLaneCenter"]
        self.leftLaneCenter = dist_pickle["leftLaneCenter"]    

    # just capsulate the warp image
    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size,flags=cv2.INTER_LINEAR)
        return warped

    # just capsulate the unwarp function
    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.Minv, img_size,flags=cv2.INTER_LINEAR)
        return unwarped

    # just capsulate the undistort function
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    # mask the area of interest and remove area left of left line, right of right line and between both lines    
    def maskAreaOfInterest(self, img,maskrange=150):

        imgshape = img.shape
        leftLaneArea = [self.leftLaneCenter-maskrange, self.leftLaneCenter+maskrange]
        rightLaneArea = [self.rightLaneCenter-maskrange, self.rightLaneCenter+maskrange]

        # remove left area
        contours = np.array( [ [0,0], [leftLaneArea[0],0], [leftLaneArea[0],imgshape[0]], [0,imgshape[0]] ] )
        cv2.fillPoly(img, pts =[contours], color=(0,0,0))

        # remove right area
        contours = np.array( [ [imgshape[1],0], [rightLaneArea[1],0], [rightLaneArea[1],imgshape[0]], [imgshape[1],imgshape[0]] ] )
        cv2.fillPoly(img, pts =[contours], color=(0,0,0))

        # remove inner area
        contours = np.array( [ [leftLaneArea[1],0], [rightLaneArea[0],0], [rightLaneArea[0],imgshape[0]], [leftLaneArea[1],imgshape[0]] ] )
        cv2.fillPoly(img, pts =[contours], color=(0,0,0))

        return img

    # convert rgb image to a pure black/white image
    def rgbConvertToBlackWhite(self, rgb, thresh=10):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        s_binary = np.zeros_like(r)
        s_binary[(r >= thresh)] = 255

        return (np.logical_or(r,b)*255)


    def contrastIncrease(self,img, factor=1.0,adaptive=False):

        return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = 0
        hsv[:,:,1] = 0

        img_v = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        img_g = cv2.cvtColor(img_v, cv2.COLOR_RGB2GRAY)

        if adaptive == False:
            ret,mask = cv2.threshold(img_g, 170, 255, cv2.THRESH_BINARY)
        else:
            mask = cv2.adaptiveThreshold(img_g,170,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

        output = cv2.bitwise_and(hsv, hsv, mask = mask)

        img_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)


        img_yuv = cv2.cvtColor(img_output, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        #img_yuv[:,:,2] = img_yuv[:,:,2]*factor
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return img_output


# just a function for debug purposes
def testLine():
    global adaptive
    testFrame = Frame()
    testFrame.initializeCamera()

    #testFrame.loadImageFromFile('../docu/img_overlay_586.png')
    testFrame.loadImageFromFile('../test_images/img_temp_1.png')
    testFrame.modifiedImg = testFrame.camera.contrastIncrease(testFrame.currentImg,1.4,False)
    testFrame.processCurrentFrame()
    testFrame.displayCurrentImage()



toggle = True
adaptive = False


def videotest():
    from moviepy.editor import VideoFileClip
    global adaptive

    testFrame = Frame()
    testFrame.initializeCamera()


    #cap = cv2.VideoCapture('../test_videos/harder_challenge_video.mp4')
    #out = cv2.VideoWriter('c:/temp/harder_challenge_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 28.0, (1280,720))    

    cap = cv2.VideoCapture('../test_videos/project_video.mp4')
    out = cv2.VideoWriter('c:/temp/project_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 24.0, (1280,720))    

    i = 0
    counter = 0 
    while(cap.isOpened()):
        ret, frame = cap.read()
        counter += 1

        if counter < 1000:
            continue


        # check for corrupted frames and drop them if necessary
        if frame == None:
            break

        if frame.shape == None:
            break

        if frame.shape[0]==0 or frame.shape[1]==0:
            break

        # if no corrupted frame was detected the     
        testFrame.currentImg = frame

        if adaptive==False:
            testFrame.modifiedImg = testFrame.camera.contrastIncrease(testFrame.currentImg,1.4,False)
        else:
            testFrame.modifiedImg = testFrame.camera.contrastIncrease(testFrame.currentImg,1.4,True)

        # finally process the frame 
        testFrame.processCurrentFrame(counter)
        
        # and store the resulting annotated overlay frame 
        #out.write(testFrame.getOverlayImage())


        # save single frames on left button click for debugging purposes
        fig = plt.figure(1)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        if (toggle == True):
            #print("Toggle activated")
            i = i + 1
            #cv2.imwrite('../temp_images/img_temp_' + str(i) + '.png', frame)
            cv2.imwrite('../temp_images_1/img_overlay_' + str(i) + '.png', testFrame.getOverlayImage())

        else:
            #print("Toggle deactivated")
            print()

        cv2.setMouseCallback('video', onclick)        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

videotest()
#testLine()
