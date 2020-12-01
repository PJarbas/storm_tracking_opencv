import argparse
import cv2
import imutils

"""
This script use the hsv color space to tracking
"""

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-exp", "--experiment", help="experiment number to run", type=int)
args = vars(ap.parse_args())

# define the lower and upper boundaries of the color
# in the HSV color space, then initialize the
# list of tracked points

experiment = {
    1: [(0, 2, 39), (180, 186, 255)],
    2: [(0, 2, 210), (117, 36, 255)],
    3: [(0, 2, 224), (117, 36, 255)],
    4: [(0, 2, 20), (117, 36, 255)],
    5: [(64, 0, 190), (172, 100, 245)],
    6: [(111, 0, 186), (255, 255, 255)],
    7: [(88, 0, 195), (255, 255, 255)],
}
limLower, limUpper = experiment[args.get("experiment")]

cap = cv2.VideoCapture(args["video"])

w = int(cap.get(3))
h = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

# keep looping
while cap.isOpened():
    # grab the current frame
    (grabbed, frame) = cap.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = cv2.resize(frame, (w, h), cv2.INTER_LANCZOS4)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, limLower, limUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if (radius < 300) & (radius > 10):
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # write the flipped frame
    out.write(frame)

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the cap and close any open windows
cap.release()
out.release()
cv2.destroyAllWindows()
