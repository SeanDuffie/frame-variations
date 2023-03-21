""" overlay.py
"""
import logging
from tkinter import filedialog
import numpy as np
import cv2


class VidCompile:
    """ Compiles an input video into one overlayed image """
    def __init__(self, path="") -> None:
        fmt_main = "%(asctime)s | VidCompile:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        # Obtain frames to go into the overlayed image
        self.frame_arr = []
        self.read_video(start=0, stop=-1, step=1)
        # Debugging code with images I made in MSpaint
        # self.frame_arr.append(cv2.imread("./overlay/box1.png", cv2.IMREAD_UNCHANGED))
        # self.frame_arr.append(cv2.imread("./overlay/box2.png", cv2.IMREAD_UNCHANGED))
        # self.frame_arr.append(cv2.imread("./overlay/box3.png", cv2.IMREAD_UNCHANGED))
        # self.frame_arr.append(cv2.imread("./overlay/box4.png", cv2.IMREAD_UNCHANGED))

        # Determine Maximum Brightness Threshold
        self.thresh = 127
        self.choose_thresh(self.frame_arr[0])
        cv2.destroyAllWindows()

        # Generate initial background image
        self.output = np.zeros(self.frame_arr[0].shape, dtype=np.uint8)
        self.output.fill(self.thresh+50)

        # Overlay each of the selected frames onto the output image
        for i, im in enumerate(self.frame_arr):
            self.overlay(im)
            logging.info("Frame %d/%d overlayed...", i+1, len(self.frame_arr))
            cv2.imshow("output", self.output)
            cv2.waitKey(1)

        # Display the final results and output to file
        # cv2.imshow("output", self.output)
        cv2.waitKey()
        cv2.imwrite("./overlay/final-long.png", self.output)

    def read_video(self, start=0, stop=-1, step=1):
        """ Read in individual frames from the video

            Inputs:
            - start: int marking the first frame of the video to include
            - stop: int marking the last frame to include
            - step: int representing the amount of frames to skip in between includes
            Outputs:
            - None
        """
        logging.debug("Reading video...")
        cap = cv2.VideoCapture("./Basler/Basler.mp4")

        # Check if camera opened successfully
        if cap.isOpened() is False:
            logging.error("Error opening video stream or file")

        # Read until video is completed or stop is reached
        c = 0
        while cap.isOpened():
            ret, frame = cap.read()     # Capture frame-by-frame
            if ret is True and c >= start and (c <= stop or stop == -1):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame_arr.append(frame)
            else:
                break
            c+=step

        # Debug for image errors
        if len(self.frame_arr) == 0:
            logging.warning("Issue Reading Video...")
        else:
            logging.debug("Video length = %d frames", len(self.frame_arr))

        cap.release()

    def choose_thresh(self, img) -> None:
        """ Decide on what threshold to apply on the image
            Anything above the threshold will be considered background and ignored

            Inputs:
            - img: numpy image array, usually is the first frame of the video, but can be key frame
            Outputs:
            - None
        """
        # Is the input image grayscale already? If not, convert it
        # gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gry = img

        # Loop until the user confirms the threshold value from the previews
        while True:
            # Generate thresholds
            ret, edit = cv2.threshold(gry,self.thresh,255,cv2.THRESH_TOZERO_INV)
            ret, binary = cv2.threshold(gry,self.thresh,255,cv2.THRESH_BINARY)

            # Show preview
            cv2.imshow("image", edit)
            cv2.imshow("binary", binary)

            # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
            Key = cv2.waitKeyEx()
            if Key == 113:
                break
            if Key == 2424832:          # Left arrow, jump down
                self.thresh -= 5
            elif Key == 2621440:        # Down arrow, step down
                self.thresh -= 1
            elif Key == 2490368:        # Up arrow, step up
                self.thresh += 1
            elif Key == 2555904:        # Right arrow, jump up
                self.thresh += 5

            # Enforce bounds and debug
            if self.thresh > 255:
                self.thresh = 255
            elif self.thresh < 0:
                self.thresh = 0
            logging.info("Current Threshold = %d", self.thresh)

    def overlay(self, img):
        """ Overlay an image onto the background
            This chooses the darker pixel for each spot of the two images
            Right now it is for grayscale images, but the commented out code can be used for color
        """
        r,c = self.output.shape
        for y in range(r):
            for x in range(c):
                if img[y,x] <= self.thresh and img[y,x] < self.output[y,x]:
                    self.output[y,x] = img[y,x]
                # if img[y,x,1] <= self.thresh and img[y,x,1] < self.output[y,x,1]:
                #     self.output[y,x,1] = img[y,x,1]
                # if img[y,x,2] <= self.thresh and img[y,x,2] < self.output[y,x,2]:
                #     self.output[y,x,2] = img[y,x,2]


if __name__ == "__main__":
    ov = VidCompile()
