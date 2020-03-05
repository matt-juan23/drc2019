#!/usr/bin/env python3
'''
DRC Vision Code

Combining blue and yellow to the inner most points
'''
import cv2, math, maestro, time
import numpy as np

def nothing(x):
    pass

def find_contour(mask, colour):
    # find the contours of a given mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coord = None
    
    if contours:
        #only consider the biggest contour, this is helps filter out the noise  
        #biggest_cont = max(contours, key=cv2.contourArea)
        biggest_conts = [cont for cont in contours if cv2.contourArea(cont) > LINE_CONTOUR_AREA - 150]
        blue_coords = []
        yellow_coords = []
        for cont in biggest_conts:
            cv2.drawContours(frame, cont, -1, (0,255,0), 3)
            # draw the circle on the left and right most points on the curves
            if colour == 'blue':
                # right of the blue line
                coord = tuple(cont[cont[:,:,0].argmax()][0])
                #cv2.circle(frame, coord, 10, (0,0,0), -1)
                blue_coords.append(coord)
            else:
                # left of the yellow line
                coord = tuple(cont[cont[:,:,0].argmin()][0])
                #cv2.circle(frame, coord, 10, (0,0,0), -1)
                yellow_coords.append(coord)
        if blue_coords and colour == 'blue':
            blue_coords.sort(key=lambda x:x[0])
            # right of the blue line
            coord = blue_coords[-1]
            cv2.circle(frame, coord, 10, (0,0,0), -1)
        elif yellow_coords and colour == 'yellow':
            # left of the yellow line
            yellow_coords.sort(key=lambda x:x[0])
            coord = yellow_coords[0]
            
            cv2.circle(frame, coord, 10, (0,0,0), -1)
        '''
        if cv2.contourArea(biggest_cont) > LINE_CONTOUR_AREA:
            cv2.drawContours(frame, biggest_cont, -1, (0,255,0), 3)
        
            #One important thing we can find about this contour is its centroid, this can be
            #very good as for example we can tell the car to steer to the point between the
            #two lines centroids
            #first thing we do is find the moments of the centroid, moments is something to
            #do with spacial distribution
            M = cv2.moments(biggest_cont)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            #draw a circle at the centroid to show where it is
            #to draw a circle use cv2.circle(image, (x, y), radius, colour, thickness)
            cv2.circle(frame, (cX, cY), 10, (0, 0, 255), -1)
            
            # draw the circle on the left and right most points on the curves
            if colour == 'blue':
                # right of the blue line
                coord = tuple(biggest_cont[biggest_cont[:,:,0].argmax()][0])
                cv2.circle(frame, coord, 10, (0,0,0), -1)
            else:
                # left of the yellow line
                coord = tuple(biggest_cont[biggest_cont[:,:,0].argmin()][0])
                cv2.circle(frame, coord, 10, (0,0,0), -1)
        '''
    if coord != None:
        # return the interger coordinates of the point we want to find the middle of
        return tuple(map(int, coord)) # most left and most right
        #return (cX, cY) # middle of contour
        #return tuple(np.mean((coord, (cX, cY)), axis=0)) # middle of above 2 coordinates
    else:
        # return None if there is no instance of the colour
        return None

def find_obstacle(mask, blue_coord, yellow_coord):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find all obstacles with a big enough contour length
    obstacles = [obs for obs in contours if cv2.arcLength(obs, True) > OBSTACLE_CONTOUR_LENGTH]
    if obstacles:
        coordinates = []
        #cv2.drawContours(frame, obstacles, -1, (0,210,100), 3)

        for obs in obstacles:
            # find the left and right most point on contour
            leftmost = tuple(obs[obs[:,:,0].argmin()][0])
            cv2.circle(frame, leftmost, 10, (250,250,250), -1)

            rightmost = tuple(obs[obs[:,:,0].argmax()][0])
            cv2.circle(frame, rightmost, 10, (250,250,250), -1)

            # find bottom of contour
            bottom = obs[obs[:,:,1].argmax()][0][1]

            # if the bottom is high enough, add the coordinates to the list
            #if bottom < HEIGHT - OBSTACLE_PASSED:
            coordinates.append(leftmost)
            coordinates.append(rightmost)

        # sort the coordinates by their x value
        coordinates.sort(key=lambda x:x[0])

        '''if coordinates:
            # don't want the program to pick up the distance between the obstacle and the edge of the screen
            if blue_coord == BLUE_MISSING_COORD:
                blue_coord = coordinates[0]
            if yellow_coord == YELLOW_MISSING_COORD:
                yellow_coord = coordinates[-1]'''
                
        if blue_coord[0] > yellow_coord[0]:
            yellow_coord = YELLOW_MISSING_COORD
        elif yellow_coord[0] < blue_coord[0]:
            blue_coord = BLUE_MISSING_COORD

        # insert the blue and yellow coordinates into the list
        coordinates.insert(0, blue_coord)
        coordinates.append(yellow_coord)
        return calc_distances(coordinates)
        
    # if there are no obstacles, return the coordinates given as input
    return (blue_coord, yellow_coord)

def calc_distances(coordinates):
    if coordinates[0][0] > coordinates[1][0]:
        coordinates[0] = coordinates[1]
    elif coordinates[-1][0] < coordinates[-2][0]:
        coordinates[-1] = coordinates[-2]
    dists = []

    # check the distances between the spaces in the track
    # increment by 2 so only check the spaces and not the width of the obstacle
    for i in range(0, len(coordinates), 2):
        #dist = math.hypot(coordinates[i][0] - coordinates[i+1][0], coordinates[i][1] - coordinates[i+1][1])
        # find the distance between the left point and right point's x value
        dist = abs(coordinates[i][0] - coordinates[i+1][0])
        dists.append(dist)

    # find the largest distance and its index in the list
    largest_dist = max(dists)
    #print(dists)
    index = dists.index(largest_dist) * 2
    #print(index)

    # return the corresponding left and right coordinates
    return (coordinates[index], coordinates[index+1])


def draw_center(left_coord, right_coord):
    # draws the center line
    midpoint_x = int((left_coord[0] + right_coord[0]) / 2)    
    midpoint_y = int((left_coord[1] + right_coord[1]) / 2)

    # Draw the line and circle
    cv2.line(frame, (int(WIDTH/2), HEIGHT), (midpoint_x, midpoint_y), (230, 100, 255), 10)
    cv2.circle(frame, (midpoint_x, midpoint_y), 10, (0, 0, 255), -1)       

def drive_car(angle):
    # probably just go 10, 20, 30, 40
    power = 0
    if -10 <= angle <= 10:
        servo.setTarget(0, 6000)
        power = 5550 - 200
    elif -30 <= angle < -10:
        servo.setTarget(0, 6900)
        power = 5600 - 80
    elif 10 < angle <= 30:
        servo.setTarget(0, 5100)
        power = 5600 - 80
    elif angle < -30:
        servo.setTarget(0, 7500)
        power = 5600
    elif angle > 30:
        servo.setTarget(0, 4500)
        power = 5600
    servo.setTarget(1, power)
    

def calc_angle(center_x, center_y):
    angle = 90
    if (center_x < WIDTH/2):
        #left side of screen
        angle = -90 + math.degrees(math.atan(abs((center_y-HEIGHT)/(center_x-WIDTH/2))))
    elif (center_x > WIDTH/2):
        #right side of screen
        angle = 90 - math.degrees(math.atan(abs((center_y-HEIGHT)/(center_x-WIDTH/2))))
    #print(angle)
    return angle

def finish_func():
    start = finish = time.time()
    while (1):
        finish = time.time()
        if finish - start > 0.5:
            break

# setting default maestro values
# 0 is for turning
# 1 is for moving
servo = maestro.Controller()
servo.setTarget(0,6000)	

# Open the cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Check that both cameras opened successfully
if (cap0.isOpened() == False or cap1.isOpened() == False): 
    print("Error opening video stream or file")
    exit(0)

WIDTH = int(cap0.get(3) * 2)
HEIGHT = int(cap0.get(4))
CROP = 180
h = HEIGHT - (int(HEIGHT/2)-CROP)
size = (WIDTH, h)

# arrays for the hsv values of the colours
BLUE_MIN = np.array([98, 65, 126])
BLUE_MAX = np.array([135, 248, 205])

YELLOW_MIN = np.array([21, 51, 155])
YELLOW_MAX = np.array([45, 147, 255])

GREEN_MIN = np.array([40, 40, 135])
GREEN_MAX = np.array([87, 100, 210])

PURPLE_MIN = np.array([135, 80, 0])
PURPLE_MAX = np.array([185, 255, 255])
#old red
RED_MIN = np.array([0, 182, 54])
RED_MAX = np.array([10 ,255,255])
#RED_MIN = np.array([0, 141, 23])
#RED_MAX = np.array([5 ,252,158])
# constants
OBSTACLE_CONTOUR_LENGTH = 200
LINE_CONTOUR_AREA = 600
OBSTACLE_PASSED = 70
BLUE_MISSING_COORD = (0, int(HEIGHT/2))
YELLOW_MISSING_COORD = (WIDTH, int(HEIGHT/2))
BOTH_MISSING_COORD = (int(WIDTH/2), int(HEIGHT/2))

# NEED THE SIZE VARIABLE TO BE THE SIZE OF THE FRAME
# NEED TO CALCULATE THE HEIGHT AS IT MIGHT CHANGE LATER
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('before_semi.avi', fourcc, 20.0, size)
nh_out = cv2.VideoWriter('nh_before_semi.avi', fourcc, 20.0, size)

while(1):
    # Take each frame
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # no highlight frame
    nh_frame = np.concatenate((frame1, frame0), axis=1)
    # highlighted frame
    frame = np.concatenate((frame1, frame0), axis=1)
    
    # check if either recording has failed
    if not ret0 or not ret1:
        print("Cannot read any frames to save")
        break

    # crop top area of image
    # the more you take away, the more vision the droid gets
    nh_frame = nh_frame[int(HEIGHT/2)-CROP:HEIGHT, 0:WIDTH]
    frame = frame[int(HEIGHT/2)-CROP:HEIGHT, 0:WIDTH]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask_yellow = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
    mask_blue = cv2.inRange(hsv, BLUE_MIN, BLUE_MAX)
    mask_green = cv2.inRange(hsv, GREEN_MIN, GREEN_MAX)
    mask_purple = cv2.inRange(hsv, PURPLE_MIN, PURPLE_MAX)
    mask_red = cv2.inRange(hsv, RED_MIN, RED_MAX)
    '''
    # Combine both masks using the bitwise_or function
    mask = cv2.bitwise_or(mask_yellow, mask_blue) 

    # Apply color to the mask
    color_filter = cv2.bitwise_and(frame, frame, mask=mask)
    green_filter = cv2.bitwise_and(frame, frame, mask=mask_green)
    purple_filter = cv2.bitwise_and(frame, frame, mask=mask_purple)
    '''  

    #Looking at the filtered image, you might notice there is random small places
    #that are being detected as the colour and areas inside the colour that are not
    #We can use the erode and dilate functions to help filter out noise and smooth images
    #First we create an element to be used for these operations, the bigger the element
    #the larger the effect it will have on the image, we are just going to make a basic
    #square element by making an array of 1's, the size of this array corresponds to the
    #size of the element
    kernel = np.ones((5, 5), np.uint8)

    # erode followed by dilate
    # basically clears most of the noise
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    obs_mask = cv2.bitwise_or(mask_purple, mask_red)

    # find the blue and yellow contour coordinate
    blue_coord = find_contour(mask_blue, 'blue')
    yellow_coord = find_contour(mask_yellow, 'yellow')

    # check if any of the colours are missing
    if blue_coord == None and yellow_coord == None:
        blue_coord = yellow_coord = BOTH_MISSING_COORD
    elif blue_coord == None:
        blue_coord = BLUE_MISSING_COORD
    elif yellow_coord == None:
        yellow_coord = YELLOW_MISSING_COORD

    # set the left and right coord which the car will drive towards
    left_coord, right_coord = find_obstacle(mask_purple, blue_coord, yellow_coord)
    #left_coord, right_coord = find_obstacle(obs_mask, blue_coord, yellow_coord)

    # not sure if we need this. the car will drive towards it anyway and the finish line is rather thin so its hard to pick up
    #find_finish(mask_green) 
    finish, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if finish:
        biggest_finishes = [cont for cont in finish if cv2.contourArea(cont) > LINE_CONTOUR_AREA]
        finish_coords = []
        finish_area = 0
        for finish in biggest_finishes:
            finish_area += cv2.contourArea(finish)
            leftmost = tuple(finish[finish[:,:,0].argmin()][0])
            rightmost = tuple(finish[finish[:,:,0].argmax()][0])
            finish_coords.append(leftmost)
            finish_coords.append(rightmost)
        if biggest_finishes:
            finish_coords.sort(key=lambda x:x[0])
            left_green = finish_coords[0][0]
            right_green = finish_coords[-1][0]
            #biggest_finish = max(finish, key=cv2.contourArea)
            '''
            if cv2.contourArea(biggest_finish) > LINE_CONTOUR_AREA:
                cv2.drawContours(frame, biggest_finish, -1, (255,255,0), 3)
                M = cv2.moments(biggest_finish)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                
                if ((WIDTH/2) - 80 < cX < (WIDTH/2) + 80) and ((HEIGHT/2) - 80 < cY < (HEIGHT/2) + 80):
                    print("found green")
                    time.sleep(0.5)
                    break
            '''
            if finish_area > LINE_CONTOUR_AREA * 6 and right_green - left_green > WIDTH//2:
                #print ("found green")t
                # find top of contour
                top = finish[finish[:,:,1].argmin()][0][1]
                if top <= HEIGHT:
                    print("found green")
                    finish_func()
                    break
                    
    # draw the center line
    draw_center(left_coord, right_coord)

    # Calculate the angle
    center_x, center_y = tuple(np.mean((left_coord, right_coord), axis=0))
    angle = calc_angle(center_x, center_y)

    # call all the maestro code that drives the car
    drive_car(angle)

    # display the windows with their respective mask
    cv2.imshow('frame', frame)
    
    # save the output to a file
    out.write(frame);    
    nh_out.write(nh_frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
# reset the movement and turning of the maestro
servo.setTarget(0, 6000)
servo.setTarget(1, 6000)

# close the cameras and servo
servo.close()
out.release()
cap0.release()
cap1.release()
cv2.destroyAllWindows()
