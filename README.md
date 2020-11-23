# storm_tracking_opencv
Using opencv to tracking storms from videos inputs

usage:

Tracking with hsv and color space method
````bash
 python storm_tracking_hsv.py --video data/storm1.mp4 --experiment 1
````

To get the HSV range colors use
````bash
python range-detector.py --filter HSV --image /path/to/image.png
````