class Object():
    def __init__(self,counter,number):
        self.detected = False
        self.location = None
        self.left_upper_y = -1
        self.left_upper_x = -1
        self.right_lower_y = -1
        self.right_lower_x = -1
        self.frameCounter = counter
        self.objectNumber = number
        self.numberOfOccurances = 0
        self.gracePeriod = False
        self.detectionThreshold = 6

        self.objectHistory = []

    def setLocation(self,l_upper_y , l_upper_x , r_lower_y, r_lower_x):
        self.left_upper_y = l_upper_y
        self.left_upper_x = l_upper_x
        self.right_lower_y = r_lower_y
        self.right_lower_x = r_lower_x
        self.validate()

        #print("Value of object in pixels " , self.getVolume())


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

        if self.numberOfOccurances < self.detectionThreshold:
            self.detected = False
            return False 


        return True            

    def getInfo(self):
        id = self.getID() + " - "
        result = id + "Left_Upper_y: " + str(self.left_upper_y) + " , Left_Upper_x: " + str(self.left_upper_x) 
        result = result + " , Right_Lower_y: " + str(self.right_lower_y) + " , Right_Lower_x: " + str(self.right_lower_x) 
        return result

    def getID(self):
        return "ID_" + str(self.frameCounter) + "_" + str(self.objectNumber)


    def initNextFrame(self):
        self.detected = False

        if self.numberOfOccurances >0: 
            self.numberOfOccurances-=1

            if len(self.objectHistory) >= 5:
                self.objectHistory.pop(0)

        if self.numberOfOccurances < 24:
            self.gracePeriod = False




    def mergeObject(self,objectToMerge):

        if self.frameCounter < objectToMerge.frameCounter:
            #self.frameCounter = objectToMerge.frameCounter
            #self.objectNumber = objectToMerge.objectNumber
            self.left_upper_y = objectToMerge.left_upper_y
            self.left_upper_x = objectToMerge.left_upper_x
            self.right_lower_y = objectToMerge.right_lower_y
            self.right_lower_x = objectToMerge.right_lower_x

        self.numberOfOccurances+=2

        if self.numberOfOccurances > 24:
            self.gracePeriod = True
            self.numberOfOccurances = 30

        if self.numberOfOccurances >= self.detectionThreshold:
            #print("Setting object to detected " ,self.getInfo())  
            self.detected = True

        self.objectHistory.append(self.clone())

        #print("Number of occurences = " ,self.numberOfOccurances)  
        return 


    def getVolume(self):
        return (self.right_lower_x - self.left_upper_x)*(self.right_lower_y - self.left_upper_y)


    def getOverlapVolume(self, otherObject):
        x = max(self.left_upper_x , otherObject.left_upper_x)
        y = max(self.left_upper_y , otherObject.left_upper_y)

        w = min(self.left_upper_x + self.right_lower_x , otherObject.left_upper_x + otherObject.right_lower_x) - x
        h = min(self.left_upper_y + self.right_lower_y , otherObject.left_upper_y + otherObject.right_lower_y) - y

#x = max(1011,902)
#min(902+975 , 1011+1223
#       1877 , 2234


#        print("X= " , x , "Y= ", y , "W= " , w, "H= " , h)

        if w<0 or h<0: 
           return 0

        if w<min(self.left_upper_x , otherObject.left_upper_x) or w>max(self.right_lower_x , otherObject.right_lower_x):
           return 0

        if h<min(self.left_upper_y , otherObject.left_upper_y) or h>max(self.right_lower_y , otherObject.right_lower_y):
           return 0

        return abs(x-w)*abs(y-h)

    def get_Left_Upper_x_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].left_upper_x

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.left_upper_x)
            return returnValue

        return self.left_upper_x

    def get_Left_Upper_y_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].left_upper_y

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.left_upper_y)
            return returnValue
        return self.left_upper_y

    def get_Right_Lower_x_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].right_lower_x

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.right_lower_x)
            return returnValue
        return self.right_lower_x


    def get_Right_Lower_y_smoothing(self):
        if len(self.objectHistory)>0:
            value = 0
            for nr in range(0, len(self.objectHistory)):            
                value += self.objectHistory[nr].right_lower_y

            returnValue = int(value/len(self.objectHistory))
            #print("Returning " , returnValue , " instead of " , self.right_lower_y)
            return returnValue
        return self.right_lower_y

    def clone(self):
        returnObject = Object(self.frameCounter,self.objectNumber)
        returnObject.detected = self.detected
        returnObject.location = self.location
        returnObject.left_upper_y = self.left_upper_y
        returnObject.left_upper_x = self.left_upper_x
        returnObject.right_lower_y = self.right_lower_y
        returnObject.right_lower_x = self.right_lower_x
        returnObject.frameCounter = self.frameCounter
        returnObject.objectNumber = self.objectNumber
        returnObject.numberOfOccurances = self.numberOfOccurances
        returnObject.gracePeriod = self.gracePeriod 
        returnObject.detectionThreshold = self.detectionThreshold

        return returnObject        
 
    def getColor(self):
        return ""

    def getDistance(self):
        return -1        