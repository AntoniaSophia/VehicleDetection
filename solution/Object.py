import numpy as np
import cv2


# Objects that can be detected 
# actually this could be different kinds of objects like cars, trucks, bikes,... 
# but only a general class is currently being implemented
class Object():
    # initial the object 
    # counter = number of the frame in which this object has been detected
    # number = number of the object to be detected in this frame
    def __init__(self,counter,number):
        self.detected = False           # indicates whether an object has been confirmed as detected 
        self.left_upper_y = -1          # left upper y value of object bounding box
        self.left_upper_x = -1          # left upper x value of object bounding box
        self.right_lower_y = -1         # right lower y value of object bounding box
        self.right_lower_x = -1         # right lower x value of object bounding box
        self.frameCounter = counter     # number of the frame in which the object has been detected
        self.objectNumber = number      # number of the object which has been detected in this frame
        self.numberOfOccurances = 0     # a counter which counts the number of occurences of this object
        self.gracePeriod = False        # in case an object has a stable detection it shall not happen to disappear suddenly in case it fails in one frame
        self.detectionThreshold = 6     # threshold of the minimal number of subsequent occurences of an object before it is confirmed as existing

        self.objectHistory = []         # history of latest found objects in order to smoothen the bounding boxes and avoid "jumping"

        self.color = ""
        self.relativeDistance = -1
        self.relativeSpeed = -1
        self.historyLength = 12

    # set the location of the object (all 4 coordinates of the bounding boxes)
    def setLocation(self,l_upper_y , l_upper_x , r_lower_y, r_lower_x):
        self.left_upper_y = l_upper_y
        self.left_upper_x = l_upper_x
        self.right_lower_y = r_lower_y
        self.right_lower_x = r_lower_x
        self.validate()

        #print("Value of object in pixels " , self.getVolume())

    # validate whether an object is a "real" object or just a potential real object
    def validate(self):
        if self.left_upper_y == -1:
            self.detected = False
            return False 
        if self.left_upper_x == -1:
            self.detected = False
            return False 
        if self.right_lower_y == -1:
            self.detected = False
            return False 
        if self.right_lower_y == -1:
            self.detected = False
            return False 

        if self.getVolume() < 1000:
            self.detected = False
            return False 

        #threshold of the minimal number of subsequent occurences of an object before it is confirmed as existing
        if self.numberOfOccurances < self.detectionThreshold:
            self.detected = False
            return False 


        return True            

    # just return basic information about this object
    def getInfo(self):
        id = self.getID() + " - "
        result = id + "Left_Upper_y: " + str(self.left_upper_y) + " , Left_Upper_x: " + str(self.left_upper_x) 
        result = result + " , Right_Lower_y: " + str(self.right_lower_y) + " , Right_Lower_x: " + str(self.right_lower_x) 
        return result

    # just return the ID of this object
    def getID(self):
        return "ID_" + str(self.frameCounter) + "_" + str(self.objectNumber)

    # initialize an object before a next frame is processed - this is an essential part of object tracking!!
    def initNextFrame(self):
        # no mercy !! every object has to proove existance again!
        self.detected = False

        # no mercy !! reduce the occurance counter
        self.numberOfOccurances-=1

        #in case we have more than historyLength historic object data just remove the latest one
        if len(self.objectHistory) >= self.historyLength:
            self.objectHistory.pop(0)

        # ok, some mercy......in case an object has been detected subsequentally over 24 frames it gets some mercy.....
        # else remove the mercy.....
        if self.numberOfOccurances < 24:
            self.gracePeriod = False

        #print("Number of occurance counter: " , self.numberOfOccurances)


    # this is also an essential part of object tracking: merge two identical objects
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


    # return the volume in pixels of this object
    def getVolume(self):
        return (self.right_lower_x - self.left_upper_x)*(self.right_lower_y - self.left_upper_y)

    # return the overlap volume of two objects
    def getOverlapVolume(self, otherObject):
        x = max(self.left_upper_x , otherObject.left_upper_x)
        y = max(self.left_upper_y , otherObject.left_upper_y)

        w = min(self.left_upper_x + self.right_lower_x , otherObject.left_upper_x + otherObject.right_lower_x) - x
        h = min(self.left_upper_y + self.right_lower_y , otherObject.left_upper_y + otherObject.right_lower_y) - y

        if w<0 or h<0: 
           return 0

        if w<min(self.left_upper_x , otherObject.left_upper_x) or w>max(self.right_lower_x , otherObject.right_lower_x):
           return 0

        if h<min(self.left_upper_y , otherObject.left_upper_y) or h>max(self.right_lower_y , otherObject.right_lower_y):
           return 0

        return abs(x-w)*abs(y-h)

    # return the left_upper_x coordinate in a smoothened way
    def get_Left_Upper_x_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].left_upper_x

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.left_upper_x)
            return returnValue

        return self.left_upper_x

    # return the left_upper_y coordinate in a smoothened way
    def get_Left_Upper_y_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].left_upper_y

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.left_upper_y)
            return returnValue
        return self.left_upper_y

    # return the right_lower_x coordinate in a smoothened way
    def get_Right_Lower_x_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].right_lower_x
            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.right_lower_x)
            return returnValue
        return self.right_lower_x


    # return the right_lower_y coordinate in a smoothened way
    def get_Right_Lower_y_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].right_lower_y

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.right_lower_y)
            return returnValue
        return self.right_lower_y

    # return the right_lower_y coordinate in a smoothened way
    def get_RelDistance_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].relativeDistance

            returnValue = round(value/len(self.objectHistory),2)
            print("Returning " , returnValue , " instead of " , self.relativeDistance)
            return returnValue
        return self.relativeDistance

    # clone an object
    def clone(self):
        returnObject = Object(self.frameCounter,self.objectNumber)
        returnObject.detected = self.detected
        returnObject.left_upper_y = self.left_upper_y
        returnObject.left_upper_x = self.left_upper_x
        returnObject.right_lower_y = self.right_lower_y
        returnObject.right_lower_x = self.right_lower_x
        returnObject.frameCounter = self.frameCounter
        returnObject.objectNumber = self.objectNumber
        returnObject.numberOfOccurances = self.numberOfOccurances
        returnObject.gracePeriod = self.gracePeriod 
        returnObject.detectionThreshold = self.detectionThreshold
        returnObject.color = self.color
        returnObject.relativeDistance = self.relativeDistance
        returnObject.relativeSpeed = self.relativeSpeed

        return returnObject        
 
    # return the color of an object
    def getColor(self):
        return self.color

    # return the relative distance of an object to the ego vehicle
    def getRelDistance(self):
        return self.relativeDistance       

    # return the relative speed of an object to the ego vehicle
    def getRelSpeed(self):
        return self.relativeSpeed                