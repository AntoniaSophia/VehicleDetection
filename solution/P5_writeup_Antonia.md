**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./../docu/car_example.png
[image2]: ./../docu/noncar_example.png
[image3]: ./../docu/hog_example_orig.png
[image4]: ./../docu/hog_example_YCrCb.png
[image5]: ./../docu/hog_example_channel_0.png
[image6]: ./../docu/hog_example_channel_1.png
[image7]: ./../docu/hog_example_channel_2.png
[image8]: ./../docu/window_search_1.png
[image9]: ./../docu/window_search_scale_0_6.png
[image10]: ./../docu/window_search_scale_1.png
[image11]: ./../docu/window_search_scale_2.png


[image20]: ./../docu/frame_1_orig.png
[image21]: ./../docu/frame_1_heatmap.png
[image22]: ./../docu/frame_1_labels.png
[image23]: ./../docu/frame_1_final_box.png

[image30]: ./../docu/frame_160_orig.png
[image31]: ./../docu/frame_160_heatmap.png
[image32]: ./../docu/frame_160_labels.png
[image33]: ./../docu/frame_160_final_box.png

[image40]: ./../docu/frame_560_orig.png
[image41]: ./../docu/frame_560_heatmap.png
[image42]: ./../docu/frame_560_labels.png
[image43]: ./../docu/frame_560_final_box.png

[image50]: ./../docu/frame_730_orig.png
[image51]: ./../docu/frame_730_heatmap.png
[image52]: ./../docu/frame_730_labels.png
[image53]: ./../docu/frame_730_final_box.png

[image60]: ./../docu/frame_760_orig.png
[image61]: ./../docu/frame_760_heatmap.png
[image62]: ./../docu/frame_760_labels.png
[image63]: ./../docu/frame_760_final_box.png

[image70]: ./../docu/frame_1000_orig.png
[image71]: ./../docu/frame_1000_heatmap.png
[image72]: ./../docu/frame_1000_labels.png
[image73]: ./../docu/frame_1000_final_box.png

[video1]: ./output_videos/project_video.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Files Submitted & Code Quality

* link to Train_Classifier.py [Training for the SVM classifier](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/Train_Classifier.py)
* link to ObjectDetection.py [Main class for ObjectDetection](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/ObjectDetection.py)
* link to Object.py [Class for identified objects](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/Object.py)
* link to Line.py [Main class for the whole pipeline](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/Line.py)
* link to Helper_Functions.py [Helper Functions for the pipeline](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/Helper_Functions.py)

* link to Jupyter monitor which shows training of the SVM classifier and heatmaps [Notebook](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/VehicleDetection.ipynb)
* link to HTML output of the Jupyter monitor which shows calibration and warp calculation [Notebook HTML](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/VehicleDetection.html)
* link to the annotated output video of the project_video.mp4 at [Project video](https://github.com/AntoniaSophia/VehicleDetection/blob/master/output_videos/project_video.mp4)
* link to the annotated output video of the challenge_video.mp4 at [Challenge video](https://github.com/AntoniaSophia/VehicleDetection/blob/master/output_videos/challenge_video.avi)

---
###Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook [Notebook](https://github.com/AntoniaSophia/VehicleDetection/blob/master/solution/VehicleDetection.ipynb) in cell ??.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car example

![car example][image1]

Non-Car example

![non-car example][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

The original image:

![HOG example orig][image3]

The same image converted into color space `YCrCb`:

![HOG converted to `YCrCb`][image4]

Now the HOG channel 0 from this image:

![HOG channel 2][image5]

Now the HOG channel 1 from this image:

![HOG channel 1][image6]

Now the HOG channel 2 from this image:

![HOG channel 2][image7]

Using this color space `YCrCb` gives much better deviations in the HOG channels than other color spaces as I found out. Differences in the above examples are clearly visible. Thus I decided to go for this color space using all HOG channels for feature extraction.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally came out with the following cell ?? ...

| Variable        | Value   | 
|:-------------:|:-------------:| 
| color_space  (# Can be RGB, HSV, LUV, HLS, YUV, YCrCb)     | 'YCrCb'       | 
| orient   (# HOG orientations)   | 9     |
| pix_per_cell   (# HOG pixels per cell)     | 8      |
| cell_per_block  (# HOG cells per block)    | 2      |
| hog_channel  (# Can be 0, 1, 2, or "ALL")    | 'ALL'       |
| spatial_size  (# Spatial binning dimensions)    | (16, 16)       |
| hist_bins (# Number of histogram bins)      | 128        |
| spatial_feat (# Spatial features on or off)     | True        |
| hist_feat (# Histogram features on or off)     | True   |
| hog_feat  (# HOG features on or off)    | True       |



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 80% training data and 20% test data (see cell ??)
Furthermore a standard feature scaler in order to prevent some features to be valued much stronger because of different input scaling. (see cell ??)
With this setup I reached an accuracy of 99.2% and decided not to play around any further (see cell ??) - the effect of a better result seems to be small compared to problems in the further steps like pipelining or false positive detection.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

- I decided to search for vehicles in the y-scale [380:600] because the the road is more or less flat.
- than I created sliding windows of size 64x64 pixels (must match the SVM training set size!!) over the whole area [380:600,0:1280] with a distance of ??
- for each window I'm searching I apply different scales in order to have a more fine-grained or rough-grained coverage of the area (see next chapter)
- for each (scaled window) I extract the HOG-features, spatial feature and histogram features 
- and finally feed the whole features of the 64x64 window into the SVM classifier in order to receive a result


All logic is contained in the function find_cars() in ObjectDetection.py lines 114-201

```
def find_cars(self,img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    find_rectangles = []

    #img = img.astype(np.float32)/255    
    draw_img = np.copy(img)


    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(
            ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    #nblocks_per_window = window // pix_per_cell - cell_per_block + 1
    #nblocks_per_window = window // pix_per_cell - cell_per_block + 1

    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):

            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the 64x64 image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            #print(spatial_features.shape)
            #print(hist_features.shape) 
            #print(hog_features.shape) 
            # Scale features and make a prediction

            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            # print(test_prediction)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),draw_color,6)
                find_rectangles.append(
                    ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                # print(find_rectangles)
    return find_rectangles
```


Here are some pictures which show this window coverage of the searching area
![window search][image8]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 8 scales (1.0 , 1.3, 1.5, 1.7, 2, 0.8, 0.6, 0.4) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

Scale 1.0
![Window Search Scale 1][image10]

Scale 2.0
![Window Search Scale 2][image11]

Scale 0.6
![Window Search Scale 0.6][image9]


The corresponding code for the vehicle detection pipeline can be found in the script ObjectDetection.py in lines 204-376 of function:
```
def processFrame(self,img,frameCounter,saveFrame=False):
```
---

- 1.Step: define the searching area in y-direction where vehicles are being assumed
- 2.Step: apply different scales to function find_cars, I used scales 1.0 , 1.3, 1.5, 1.7, 2, 0.8, 0.6, 0.4
- 3.Step: apply heatmap to all detected object bounding boxes
- 4.Step: Find final boxes from heatmap using label function
- 5.Step: extract the "raw objects" from all objects that have been found according to the heatmap
- 6.Step: now apply object plausibilization and object tracking to all "raw objects"
- 7.Step: take the bounding box of the object, resize it to 64x64 pixels and apply it on the SVC again 
- 8.Step: now use confidence of the SVM and consider all objects below the threshold 0.4 as non-detections --> sorting  out!
- 9.Step calculate the color of the object
- 10.Step Test: in case the threshold value of 0.4 has been passed --> call object tracking function


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my video [Project video](https://github.com/AntoniaSophia/VehicleDetection/blob/master/output_videos/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

Frame 1 - original image

![image20][image20]


Frame 1 - corresponding heatmap

![image21][image21]


Frame 160 - original image

![image30][image30]


Frame 160 - corresponding heatmap

![image31][image31] 

Frame 560 - original image

![image40][image40] 

Frame 560 - corresponding heatmap

![image41][image41] 


Frame 730 - original image

![image50][image50] 

Frame 730 - corresponding heatmap

![image51][image51] 

Frame 760 - original image

![image60][image60] 

Frame 760 - corresponding heatmap

![image61][image61] 


Frame 1000 - original image

![image70][image70] 

Frame 1000 - corresponding heatmap

![image71][image71] 


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
Frame 1 - labels in gray after thresholding

![image22][image22]

Frame 160 - labels in gray after thresholding

![image32][image32]

Frame 560 - labels in gray after thresholding

![image42][image42]

Frame 730 - labels in gray after thresholding

![image52][image52]

Frame 760 - labels in gray after thresholding

![image62][image62]

Frame 1000 - labels in gray after thresholding

![image72][image72]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
Frame 1 - final boxes

![image23][image23]


Frame 160 - final boxes

![image33][image33]

Frame 560 - final boxes

![image43][image43]

Frame 730 - final boxes

![image53][image53]

Frame 760 - final boxes

![image63][image63]


Frame 1000 - final boxes

![image73][image73]



More steps for removal of false positives can be found in the next chapter "Object Tracking"

### Object Tracking 

The actual object tracking is done in the function track_object() in ObjectDetection.py (line 69-97)
- 1.Step: iterate through all exisiting objects
- 2.Step: check whether there a high overlap with an already existing object
- 3.Step: In case the overlap > 500 than assume both objects are same and merge both objects
- 4.Step: if no overlap match has been found --> append this object as newly found object

```
# track an object - this is the heart function of object tracking!!
# assume newObject as the newly found object in a frame
def track_object(self,allObjects, newObject):
    existingObjectFound = False

    # 1.Step: iterate through all exisiting objects
    for object_number in range(0, len(allObjects)):
        tempObject = allObjects[object_number]

        # 2.Step: check whether there a high overlap with an already existing object
        overlap = tempObject.getOverlapVolume(newObject)
        #print("Overlap " , newObject.getInfo() , "  with existing " + tempObject.getInfo() , " : " , overlap)

        # 3.Step: In case the overlap > 500 than assume both objects are same and merge both objects
        if  overlap > 500:
            existingObjectFound = True  

            tempObject.mergeObject(newObject,self.M)

            # only merge one object (to the first matching object!) and not too all objects
            continue


    # 4.Step: if no overlap match has been found --> append this object as newly found object
    if existingObjectFound == False:
        print("Appending object: " , newObject.getInfo())
        allObjects.append(newObject)

    return allObjects
```


Furthermore I have added the following mechanism in order to avoid false positives:
- an object has be detected at least a minimal number of subsequent frames (detectionThreshold was set to 6 in class Object.py)
- in order to have a confirmed object it must have at least an overlap volume of 500 pixels, otherwise it it considered as different object


The following mergeObject function is the heart of object tracking and removal of false positives (see Object.py lines 100-156)
```
def mergeObject(self,objectToMerge,M):
	#1.Step: only merge in case the new object has a higher frame counter
	if self.frameCounter < objectToMerge.frameCounter:

	    #2.Step don't copy the object - but update the existing object!!
	    self.left_upper_y = objectToMerge.left_upper_y
	    self.left_upper_x = objectToMerge.left_upper_x
	    self.right_lower_y = objectToMerge.right_lower_y
	    self.right_lower_x = objectToMerge.right_lower_x

	    # don't merge frameCounter or objectNumber as this would create "new" objects !!!
	    #self.frameCounter = objectToMerge.frameCounter
	    #self.objectNumber = objectToMerge.objectNumber

	    #3.Step: get the middle of the bottom line of the bounding box and warp this point
	    pos = np.array((self.get_Left_Upper_x_smoothing() + 
	        (self.get_Right_Lower_x_smoothing()-self.get_Left_Upper_x_smoothing())/2,self.get_Right_Lower_y_smoothing()),dtype="float32").reshape(1, 1, -1)
	    dst1 = cv2.perspectiveTransform(pos, M).reshape(-1, 1)

	    #4.Step: get the middle of the image bottom (image is 720x1280) and warp this point
	    reference_warped = np.array((720,640),dtype="float32").reshape(1, 1, -1)
	    dst_warped = cv2.perspectiveTransform(reference_warped, M).reshape(-1, 1)

	    #5.Step: calculate the pixel distance of both points and calculate the y distance in meter space , 1 pixel ~ 30/720 meters
	    ym_per_pix = 30/720
	    distance = round(((dst_warped[1]-dst1[1])*ym_per_pix)[0],2)
	    self.relativeDistance = distance


	    # 6.Step calculate the relative speed of the object
	    if len(self.objectHistory) == self.historyLength-1:     #we have no appended this object yet --> so subtract 1 from historyLength
	        delta_s = (self.get_RelDistance_smoothing() - self.objectHistory[0].get_RelDistance_smoothing())    #difference in relative distance
	        delta_t = self.historyLength/24         # difference in delta time, assuming 24 frames per second (attention: 24fps is hardcoded here!!)
	        relativeSpeed = round((delta_s/delta_t)*3.6,2)      #relative speed is just delta distance / delta time * 3.6 (3.6 is factor to convert from m/s to km/h)
	        self.relativeSpeed = relativeSpeed
	        #print("#####################Relative Speed: " , relativeSpeed)


	#7.Step: increase the occurence counter
	self.numberOfOccurances+=2

	#8.Step: ok, distribute some mercy......in case an object has been detected subsequentally over 24 frames it gets some mercy.....
	if self.numberOfOccurances > 24:
	    self.gracePeriod = True
	    # limit the occurence counter to 30
	    self.numberOfOccurances = 30


	#9.Step: threshold of the minimal number of subsequent occurences of an object before it is confirmed as existing
	if self.numberOfOccurances >= self.detectionThreshold:
	    #print("Setting object to detected " ,self.getInfo())  
	    self.detected = True

	#10.Step: finally append the object clone to the object history......
	self.objectHistory.append(self.clone())

	return 
```




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One specific problem is that the SVM classifier has to use the identical parameters "in use" as for training. Therefor I decided to store all training parameter (including the color space) in a pickle file and reuse it.
The object tracking was relatively simple, I felt it was just straight forward programming. However there are some parameter which have not been tuned for other videos like:
- minimal subsequent occurrence threshold until an object is considered to be detected
- minimal overlap in pixel volume in order to decide whether two objects are the same
- scales being used for finding cars
- more robust SVM classifier

I'm sure my pipeline will have problems in the follwoing situations:
- videos which contain a lot of shadows (shadow has not been tested at all)
- night images have not been provided in the training set at all (training on white or red lights in the image which could be origined from a car)
- only cars can be detected, no trucks or bikes,.... (I would need more training data for the SVM classifier)
- due to the fixed search for windows in the y-scale area [380:600] my pipeline will also not work for mountainous roads
- my object tracking is adjusted most likely to slow moving objects as 
- the performance of my pipepline is bad because I have used so many scales (1.0 , 1.3, 1.5, 1.7, 2, 0.8, 0.6, 0.4) - definitely should be made much faster!!

One additional problem is when more than one cars are visible at almost the same place of even overlapping (as towards the end of the video). Distinuishing between different objects is really hard in my opinion (at least I have no idea spontanously). My object tracking just decided to "spawn" a new object in frame 925, but didn't recognize that it was an object that was already visible before...


Regarding the relative speed which I calculated: this was a nice try as all information was available (I already calculated the relative distance).
But I faced the following problems which make the relative speed information almost worthless: 
- the bounding box gets bigger in case cars are rushing into the image --> so the lower y values increases which leads to negative relative speed even if the car is slowly overtaking... :-)
- different curvatures of ego vehicle and tracked object is not calculated
- every small bouncing of the camera picture immediately leads to changing y-values of the bounding boxes and thus influences the distance calculation and speed

I'm sure the second term will get to know much techniques like kalman filter and more accurate calculation of the bounding boxes!!


Really looking forward to term 2 and I'm really happy to be part of this Udacity Nanodegress program- lot's of fun learning new stuff and somehow also proud of the results I achieved in the different projects!!



Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

