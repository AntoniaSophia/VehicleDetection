import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from Helper_Functions import *
from Object import *
import glob
import itertools
from scipy.ndimage.measurements import label

class ObjectDetection():
    def __init__(self):
        self.svmConfigFilename = None
        self.svc = None
        self.X_scaler = None
        self.orient = None
        self.pix_per_cell = -1
        self.cell_per_block = -1
        self.spatial_size = -1
        self.hist_bins = -1
        self.objectsDetected = []        
        self.hog_channel = -1
        self.color_space = None

    def __init__(self,filename):
        self.svmConfigFilename = None
        self.svc = None
        self.X_scaler = None
        self.orient = None
        self.pix_per_cell = -1
        self.cell_per_block = -1
        self.spatial_size = -1
        self.hist_bins = -1
        self.objectsDetected = []      
        self.hog_channel = -1  
        self.color_space = None

        self.loadSVMConfiguration(filename)

    def loadSVMConfiguration(self,filename):

        dist_pickle = pickle.load(open(filename, "rb"))
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["X_scaler"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]
        self.hog_channel = dist_pickle["hog_channel"]
        self.color_space = dist_pickle["color_space"]

    def track_object(self,allObjects, newObject):
        existingObjectFound = False

        for object_number in range(0, len(allObjects)):
            tempObject = allObjects[object_number]

            overlap = tempObject.getOverlapVolume(newObject)
            #print("Overlap " , newObject.getInfo() , "  with existing " + tempObject.getInfo() , " : " , overlap)
            if  overlap > 1000:
                existingObjectFound = True            
                tempObject.mergeObject(newObject)

        if existingObjectFound == False:
            print("Appending object: " , newObject.getInfo())
            allObjects.append(newObject)

        return allObjects

    def cleanup_objects(self,allObjects,counter):
        result = []
        for object_number in range(0, len(allObjects)):
            tempObject = allObjects[object_number]

            if (counter - tempObject.frameCounter) > 72:
                if tempObject.detected == False and tempObject.gracePeriod == False:
                    # remove this object
                    print()
                else:
                    result.append(tempObject)      
            else:
                result.append(tempObject)      


        return result

    # Define a single function that can extract features using hog
    # sub-sampling and make predictions
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

        i = 0
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

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

                i += 1

                #cv2.imwrite('./debug/img_temp_' + str(i) + '.jpg', subimg)

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

    def processFrame(self,img,frameCounter,saveFrame=False):
        vehicle_boxes = []
        temp_vehicle_boxes = []

        img = mpimg.imread(image)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ystart = 380
        ystop = 600

        #################################
        temp_vehicle_boxes = []
        draw_color = (255,0,0) 
        scale = 1.0
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        #################################



        # #################################
        temp_vehicle_boxes = []
        draw_color = (0,255,0) 
        scale = 1.3
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################

        # #################################
        temp_vehicle_boxes = []
        draw_color = (0,0,255) 
        scale = 1.5
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################

       # #################################
        temp_vehicle_boxes = []
        draw_color = (0,0,255) 
        scale = 2
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################

       # #################################
        temp_vehicle_boxes = []
        draw_color = (0,0,255) 
        scale = 1.7
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################


        # ystart = 360
        # ystop = 450

        # #################################
        temp_vehicle_boxes = []
        draw_color = (0,0,255) 
        scale = 0.8
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################

        # #################################
        temp_vehicle_boxes = []
        draw_color = (0,0,255) 
        scale = 0.6
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################

        # #################################
        temp_vehicle_boxes = []
        draw_color = (0,0,255) 
        scale = 0.4
        result_list = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                           self.cell_per_block, self.spatial_size, self.hist_bins)

        temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
        out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
        vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
        # #################################


        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, vehicle_boxes)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        heatmap = apply_threshold(heatmap, 10)

        # Find final boxes from heatmap using label function

        labels = label(heatmap)
        print(labels[1], 'cars found')



        foundObjects = create_Objects_from_Labels(labels,counter)

        for object_number in range(0, len(foundObjects)):
            img_temp = np.copy(img)
            img_tosearch = img_temp[foundObjects[object_number].left_upper_y:foundObjects[object_number].right_lower_y,foundObjects[object_number].left_upper_x:foundObjects[object_number].right_lower_x,:]
            subimg = cv2.resize(img_tosearch[:,:], (64, 64))
            #subimg = np.copy(img)

            ttt_features = extract_features_img(subimg, color_space='YCrCb', 
                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                cell_per_block=self.cell_per_block, 
                hog_channel=self.hog_channel, spatial_feat=True, 
                hist_feat=True, hog_feat=True)
            test_features = self.X_scaler.transform(ttt_features)
            test_prediction = self.svc.predict(test_features)
            test_confidence = self.svc.decision_function(test_features)
            print("Object found with confidence: " , test_confidence)

            if test_prediction > 0 and test_confidence > 0.5:
                self.track_object(self.objectsDetected , foundObjects[object_number])

        print("Length: Temp" , len(foundObjects))
        print("Length All: " , len(self.objectsDetected))


        #if counter%100 == 0:
        #    self.objectsDetected = self.cleanup_objects(self.objectsDetected,counter)

        if saveFrame==True:
            draw_img = draw_objects(np.copy(img), self.objectsDetected)
            mpimg.imsave('./../output/img_' + str(counter) + '.png', draw_img)


        for object_number in range(0, len(self.objectsDetected)):
            self.objectsDetected[object_number].initNextFrame()


        return self.objectsDetected


#images = glob.glob('./../video_test_data/*.jpg', recursive=True)
images = glob.glob('./../../Project_Video/*.jpg', recursive=True)
counter = 0

objectDet = ObjectDetection("../svm_cal/svc_pickle.p")



for image in images:
    counter+=1

    #if counter%5!=0:
    #    continue
    
    if counter<1000:
       continue
    #    exit()

    objectDet.processFrame(image,counter,True)


