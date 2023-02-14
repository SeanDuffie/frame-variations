""" vid_test.py
"""
from tkinter import Button, Label, Tk, filedialog
import time
import logging

import cv2
import numpy as np

class VidClass:
    """Handles video processing"""
    def __init__(self):
        self.path = self.select_vid()
        self.FRAME_ARR = self.get_vid()

    def select_vid(self):
        """ Open a file chooser dialog and allow the user to select an input image """
        select = filedialog.askopenfilename()
        if len(select) > 0:
            print(f"Selected Path = {select}")
            return select
        return "VID_20221226_155105.mp4"

    def get_vid(self):
        """ Puts all frames into an array """
        frames = []
        cap = cv2.VideoCapture(self.path)#, cv2.CAP_DSHOW)

        # Check if camera opened successfully
        if cap.isOpened() is False:
            print("Error opening video stream or file")
        
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

        print(f"Video length = {len(frames)} frames")
        
        cap.release()
        return frames

    def play_vid(self):
        """ Plays the frames in order """
        stop = False
        while not stop:
            i = 0
            while i < len(self.FRAME_ARR):
                print(f"Frame {i}, ", str(int(i/30)) + "." + str(i%30), "seconds")
                cv2.imshow('Frame', self.FRAME_ARR[i])
            # for i, frame in enumerate(self.FRAME_ARR):
            #     print(f"Frame {i}, ", str(int(i/30)) + "." + str(i%30), "seconds")
            #     cv2.imshow('Frame',frame)
                i += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(3) & 0xFF == ord('q'):
                    stop = True
                    break

    def select_frames(self, start, end, interval):
        """ Print out selected frames """
        i = start
        while i <= end:
            cv2.imwrite(f"frames/frame{i}.jpg", self.FRAME_ARR[i])
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
    vid = VidClass()

    print("Previewing Video frames...")
    vid.play_vid()

    a = int(input("start frame: "))
    b = int(input("end frame: "))
    i = int(input("interval: "))
    vid.select_frames(a, b, i)
    
    cv2.destroyAllWindows()