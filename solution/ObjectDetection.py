import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from Helper_Functions import *
import glob
import itertools
from scipy.ndimage.measurements import label

 
dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

#img = mpimg.imread('./examples/test6.jpg')
#img = mpimg.imread('./examples/test7.jpg')
#img = mpimg.imread('./examples/bbox-example-image.jpg')

# Define a single function that can extract features using hog
# sub-sampling and make predictions


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    find_rectangles = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

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

frames = []
images = glob.glob('./../video_test_data/*.jpg', recursive=True)

ystart = 400
ystop = 600

counter = 0

result_list = []
vehicle_boxes = []


def append_boxes(a,b):
    for i in range(0,len(b),1):
        a.append([b[i][0], b[i][1]])

    return a


for image in images:
    counter+=1

    #if counter%10!=0:
    #    continue
    #if counter>10:
    #    exit()
    vehicle_boxes = []
    temp_vehicle_boxes = []

    img = mpimg.imread(image)

    #################################
    temp_vehicle_boxes = []
    draw_color = (255,0,0) 
    scale = 1.0
    result_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

    temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
    out_img = draw_boxes(img, temp_vehicle_boxes, color=draw_color, thick=6)
    vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
    #################################



    # #################################
    temp_vehicle_boxes = []
    draw_color = (0,255,0) 
    scale = 1.3
    result_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

    temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
    out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
    vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
    # #################################


    # #################################
    temp_vehicle_boxes = []
    draw_color = (0,0,255) 
    scale = 0.8
    result_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

    temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
    out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
    vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
    # #################################

    # #################################
    temp_vehicle_boxes = []
    draw_color = (0,0,255) 
    scale = 1.5
    result_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

    temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
    out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
    vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
    # #################################

   # #################################
    temp_vehicle_boxes = []
    draw_color = (0,0,255) 
    scale = 2
    result_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

    temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
    out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
    vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
    # #################################

   # #################################
    temp_vehicle_boxes = []
    draw_color = (0,0,255) 
    scale = 1.7
    result_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

    temp_vehicle_boxes = append_boxes(temp_vehicle_boxes,result_list)
    out_img = draw_boxes(out_img, temp_vehicle_boxes, color=draw_color, thick=6)
    vehicle_boxes = append_boxes(vehicle_boxes , temp_vehicle_boxes)
    # #################################


    plt.imshow(out_img)
    plt.show()
 

    #exit()


    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, vehicle_boxes)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    heatmap = apply_threshold(heatmap, 10)

    # Find final boxes from heatmap using label function

    labels = label(heatmap)
    print(labels[1], 'cars found')

    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()
    #cv2.imwrite('./../output/img_' + str(counter) + '.jpg', draw_img)
    #plt.savefig('./output/img_' + str(counter) + '.jpg')


#exit()
#
#    plt.imshow(out_img)
#    plt.show()