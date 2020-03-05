#********************** Details of Code ******************************
#Author: Nikola Medimurac
#Date: 3/17/2019
#Code to help find the colour range required to filter out a specific
#colour. Can take images, videos or camera feeds as input and filter
#in several different colour spaces which are HSV, HLS, YCrCb, LAB,
#LUV, RGB and BGR.
#*********************************************************************
#*********** Instructions for how to use code ************************
#*********************************************************************
#the code can be run in multiple ways the simplest is to just run it
#like a normal python script and input the name of the image source
#and colour space when prompted. Adjust the trackbar to see what colours
#in the image fall under the range selected, you can also click on the
#image to set the trackbars to the colour of the pixel clicked. Videos and
#camera feeds can be paused on the current frame by pressing the space
#button, this has some problems though like it wont pause sometimes and
#holding it switches the pause on and off constantly
#
#The other way to run the code is to input the image source and colour
#space when calling the python script by writing
#'python findColourRange.py "imageSource" "colourSpace"'
#eg. python findColourRange.py myTestVideo.avi HLS
#if no colour space is given the default value is HSV
#*********************************************************************
import cv2
import numpy as np
import sys

#Function that checks if the input is a image, video or camera
#returns videoCapture object if a video is passed and a boolean value
#indicating if the source is a video or image
def getSourceType (imageSource):

    #Check if an image was passed
    imageStream = cv2.imread(imageSource)
    if imageStream != None:
        isVideo = False
        print("image recieved")
        return imageStream, isVideo
    
    #First check if a video was passed
    imageStream = cv2.VideoCapture(imageSource)
    if imageStream.isOpened() == True:
        isVideo = True
        print("video recieved")
        return imageStream, isVideo

    #Check if a camera was passed
    try:
        cameraNumber = int(imageSource)
        imageStream = cv2.VideoCapture(cameraNumber)
        if imageStream.isOpened() == True:
            isVideo = True
            print("camera recieved")
            return imageStream, isVideo
    except ValueError:
        isVideo = False
        
    #if input doesnt match any, return imageStream = None to return error
    isVideo = False
    imageStream = None
    print("image source could not be found! :(")
    return imageStream, isVideo

#Callback function for the mouse click, sets trackbar values to the values
#at the point clicked
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [x, y]
        cv2.setTrackbarPos('Channel 1 lower', 'trackbars', imageStackedChangeColour[y, x][0])
        cv2.setTrackbarPos('Channel 2 lower', 'trackbars', imageStackedChangeColour[y, x][1])
        cv2.setTrackbarPos('Channel 3 lower', 'trackbars', imageStackedChangeColour[y, x][2])
        cv2.setTrackbarPos('Channel 1 Upper', 'trackbars', imageStackedChangeColour[y, x][0])
        cv2.setTrackbarPos('Channel 2 Upper', 'trackbars', imageStackedChangeColour[y, x][1])
        cv2.setTrackbarPos('Channel 3 Upper', 'trackbars', imageStackedChangeColour[y, x][2])

#function to covert the image to the chosen colour space
def changeColourSpace (Image, colourSpace):
    if colourSpace == "HSV":
        changedColour = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif colourSpace == "HLS":
        changedColour = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif colourSpace == "LAB":
        changedColour = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif colourSpace == "LUV":
        changedColour = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif colourSpace == "YCC" or colourSpace == "YCRCB":
        changedColour = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    elif colourSpace == "BGR":
        changedColour = image
    elif colourSpace == "RGB":
        changedColour = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    else:
        print("invalid colour space chosen")
    return changedColour

#Trackbars need a callback function but i dont want one so make a useless function
def nothing(x):
    pass
    
#************* Main Code ******************

#First thing to do is load in the inputs
#check number of inputs from command line
if len(sys.argv) == 3:
    imageSource = sys.argv[1]
    colourSpace = sys.argv[2]

#if only 1 input is given then set color space to hsv
if len(sys.argv) == 2:
    imageSource = sys.argv[1]
    colourSpace = "HSV"

#if no inputs are given, create an input for the user to enter them
if len(sys.argv) == 1:
    imageSource = input("What is the name of your image source: ")
    colourSpace = input("What colour space do you want to filter in : ")

#change colourSpace to all captials, this avoids case sensitive inputs
colourSpace = colourSpace.upper()

#Check the inputs are valid
imageStream, isVideo = getSourceType(imageSource)

#Setup callback function for mouse click
cv2.namedWindow('Images')
cv2.resizeWindow('Images', 640, 320)
cv2.setMouseCallback("Images", mouse_click)

#Make the trackbars
cv2.namedWindow('trackbars')
cv2.createTrackbar('Channel 1 lower', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('Channel 2 lower', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('Channel 3 lower', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('Channel 1 Upper', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('Channel 2 Upper', 'trackbars', 0, 255, nothing)
cv2.createTrackbar('Channel 3 Upper', 'trackbars', 0, 255, nothing)

#Value that sets if the video is paused, by default dont pause the video
pauseVideo = False

#setup the while loop
while(1):
    #Make pressing space keep the previous image until space is pressed again
    #the code will be frozen until space is released
    if cv2.waitKey(25) & 0xFF == ord(' '):
        pauseVideo = ~pauseVideo
        #Print if it is paused or not
        if pauseVideo == False:
            print("video playing")
        else:
            print("video paused")
            
                
    #Load in the image, need to check if it is a normal image or video
    #and load it in appropriately
    if pauseVideo == False:
        if isVideo == True:
            isWorking, image = imageStream.read()
            if isWorking == False:
                print("video stream ended")
                break
        else:
            image = imageStream

    #resize the image to 640x480
    image = cv2.resize(image, (480,320))

    #convert the image to the chosen colour space
    changedColour = changeColourSpace(image, colourSpace)
    
    #Get trackbar values
    caLower = cv2.getTrackbarPos('Channel 1 lower', 'trackbars')
    cbLower = cv2.getTrackbarPos('Channel 2 lower', 'trackbars')
    ccLower = cv2.getTrackbarPos('Channel 3 lower', 'trackbars')
    caUpper = cv2.getTrackbarPos('Channel 1 Upper', 'trackbars')
    cbUpper = cv2.getTrackbarPos('Channel 2 Upper', 'trackbars')
    ccUpper = cv2.getTrackbarPos('Channel 3 Upper', 'trackbars')

    #filter for colours in the range set
    filteredImage = cv2.inRange(changedColour, (caLower, cbLower, ccLower), (caUpper, cbUpper, ccUpper))

    #apply mask on original image
    maskedImage = cv2.bitwise_and(image, image, mask = filteredImage)

    #stack the images next to each other so its easier to see
    imageStacked = np.hstack((image, maskedImage))
    imageStackedChangeColour = changeColourSpace(imageStacked, colourSpace)
    
    #display the original image and the masked image
    cv2.imshow("Images", imageStacked)        

    #This weird if statement is to make pressing q exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break  

#close all the windows before the code finishes
cv2.destroyAllWindows()
    
    
