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

        if self.numberOfOccurances < 4:
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

        if self.numberOfOccurances < 24:
            self.gracePeriod = False


    def mergeObject(self,objectToMerge):
        if self.frameCounter < objectToMerge.frameCounter:
            self.frameCounter = objectToMerge.frameCounter
            self.objectNumber = objectToMerge.objectNumber
            self.left_upper_y = objectToMerge.left_upper_y
            self.left_upper_x = objectToMerge.left_upper_x
            self.right_lower_y = objectToMerge.right_lower_y
            self.right_lower_x = objectToMerge.right_lower_x

        self.numberOfOccurances+=2

        if self.numberOfOccurances > 24:
            self.gracePeriod = True

        if self.numberOfOccurances >= 4:
            self.detected = True

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
 