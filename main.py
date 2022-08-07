import cv2 # import open cv
from tracker import * # from tracker.py import everything

# Create tracker object
tracker = EuclideanDistTracker() # this tracker takes all the bounding boxes of the objects of the object into one array

cap = cv2.VideoCapture("highway.mp4") # create a capture object to read the frames from the video with the path of the video file

# Object detection from Stable camera
# this object detection is going to extract the mooving object fron the stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # parameters : history= the longer is the history the more precise and adapt to the changes if the camera moves | varThreshold = the lower is the value the more results we get but also this is chance we get many False positive

while True: # because considering this is a video, we need to extract the frames one after an other
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # print(height, width) to get the shapes of the frame and we ll get 720; 1280

    # lets extract a region of interest : the road going "downway"
    region_of_interest = frame[340: 720, 500: 800]

    # 1. Object Detection
    mask = object_detector.apply(region_of_interest)  # create a mask : make everything we don t need black and the things that interest us in white in the region of interest
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # cleaning the mask by creating a threeshold to eliminate grey pixels of the mask and keep only the white pixels so : "under 254 don t keep it" to "high limit = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find the boundaries of the moving objects
    detections = [] # empty array and each time we find the box we re going to put them inside it
    for cnt in contours:
        # Calculate area and remove small and useless elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h]) # add the new box to the array

    # 2. Object Tracking
    boxes_IDs = tracker.update(detections)
    for box_ID in boxes_IDs:
        x, y, w, h, ID = box_ID
        cv2.putText(region_of_interest, str(ID), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) # parameters : region_of_interest, the box's ID, the position(x, y - 15) ; the font : cv2.FONT_HERSHEY_PLAIN ; color : (255, 0, 0) ; thickness: 2
        cv2.rectangle(region_of_interest, (x, y), (x + w, y + h), ( 0,0,255), 2)  # representing the "box" that surrounding each object detected with top left point x,y and right bottom point x+width, y+high ; and choosing the color red with green & blue = 0 and red= 255 : ( 0,0,255) |

    cv2.imshow("roi", region_of_interest)
    cv2.imshow("Frame", frame) # show the frame in real time
    cv2.imshow("Mask", mask) # show the mask we created to see the moving elements from the video

    key = cv2.waitKey(30) # if cv2.waitKey(0) it will wait for us to pass to the next frame ; for num diff to 0 it will wait x millisecond between each frame while "cv2.waitKey(x)"
    if key == 27: # if we want to close the video we just have to press the 27 key on keybord
        break  # break the loop

cap.release()
cv2.destroyAllWindows()