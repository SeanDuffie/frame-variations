""" vid_test.py
"""
import os
from tkinter import filedialog
import time
import logging

import cv2
import numpy as np

class VidClass:
    """Handles video processing"""
    def __init__(self, path):
        # Initial Logger Settings
        fmt_main = "%(asctime)s | main\t\t: %(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%D %H:%M:%S")
        self.path = path
        self.name = os.path.basename(self.path)
        self.FRAME_ARR = self.get_vid()

    def select_vid(self):
        """ Open a file chooser dialog and allow the user to select an input image """
        select = filedialog.askopenfilename()
        if len(select) > 0:
            logging.info("Selected Path = %s", select)
            return select
        return "VID_20221226_155105.mp4"

    def get_vid(self):
        """ Puts all frames into an array """
        logging.info("Reading video...")
        logging.info("This takes a minute or two...")
        frames = []
        cap = cv2.VideoCapture(self.path + "/" + self.name + ".mp4")#, cv2.CAP_DSHOW)

        # Check if camera opened successfully
        if cap.isOpened() is False:
            logging.info("Error opening video stream or file")
        
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret is True:
                # Display the resulting frame
                frame = self.resize_img(frame, .25)     # TODO: make resizing variable
                frames.append(frame)
            else:
                break

        logging.info(f"Video length = {len(frames)} frames")
        
        cap.release()
        return frames

    def play_vid(self):
        """ Plays the frames in order """
        logging.info("Playing video...")
        stop = False
        while not stop:
            i = 0
            while i < len(self.FRAME_ARR):
                # logging.info("Frame %d, %d.%d seconds", i, int(i/30), i%30)
                cv2.imshow('Frame', self.FRAME_ARR[i])
            # for i, frame in enumerate(self.FRAME_ARR):
            #     logging.info(f"Frame {i}, ", str(int(i/30)) + "." + str(i%30), "seconds")
            #     cv2.imshow('Frame',frame)
                i += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(3) & 0xFF == ord('q'):
                    stop = True
                    break

    def select_frames(self, start, end, interval):
        """ logging.info out selected frames """
        i = start
        while i <= end:
            cv2.imwrite(f"{self.path}/frames/frame{i}.jpg", self.FRAME_ARR[i])
            i += interval

    def resize_img(self, img, scale):
        """
        Scales the image by the ratio passed in scale
        """
        w1 = img.shape[1]
        h1 = img.shape[0]
        w2 = int(w1 * scale)
        h2 = int(h1 * scale)
        new_dim = (w2, h2)
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    path = filedialog.askdirectory()
    vid = VidClass(path)

    logging.info("Previewing Video frames...")
    vid.play_vid()

    a = -1
    b = -1
    i = -1
    while a < 0:
        try:
            a = int(input("start frame: "))
        except:
            logging.info("enter an int!")
            a = -1
    while b < 0:
        try:
            b = int(input("end frame: "))
        except:
            logging.info("enter an int!")
            b = -1
    while i < 0:
        try:
            i = int(input("interval: "))
        except:
            logging.info("enter an int!")
            i = -1
    vid.select_frames(a, b, i)
    
    cv2.destroyAllWindows()