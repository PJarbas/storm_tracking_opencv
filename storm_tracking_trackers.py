import argparse
import cv2
import imutils

"""
This script using the trackers from the opencv API to tracking
"""

# reference: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-tr", "--tracker", help="tracker algorithm name to run", type=str, default="mosse")
args = vars(ap.parse_args())

# We define a bounding box containing the object
# for the first frame and initialize the tracker
# with the first frame and the bounding box.
# Finally, we read frames from the video and just update the tracker
# in a loop to obtain a new bounding box for the current frame.

trackers = {
    'BOOSTING': cv2.TrackerBoosting_create(),
    'MIL': cv2.TrackerMIL_create(),
    'KCF': cv2.TrackerKCF_create(),
    'TLD': cv2.TrackerTLD_create(),
    'MEDIANFLOW': cv2.TrackerMedianFlow_create(),
    'GOTURN': cv2.TrackerGOTURN_create(),
    'MOSSE': cv2.TrackerMOSSE_create(),
    'CSRT': cv2.TrackerCSRT_create(),
}


def draw_box(frame, bbox):
    x, y, w, h = list(map(int, bbox))
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)


tracker = trackers[args["tracker"].upper()]

cap = cv2.VideoCapture(args["video"])

(success, frame) = cap.read()

# Select ROI in the First frame
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

# keep looping
while cap.isOpened():

    # get timer to compute FPS
    timer = cv2.getTickCount()

    # grab the current frame
    (success, frame) = cap.read()

    success, bbox = tracker.update(frame)

    if success:
        draw_box(frame, bbox)
    else:
        cv2.putText(frame, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 230), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(frame, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not success:
        break

    # show the frame to our screen
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the cap and close any open windows
cap.release()
cv2.destroyAllWindows()
