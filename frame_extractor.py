""" vid_test.py
"""
import logging
import os
from tkinter import filedialog
from typing import Any

import cv2
import numpy.typing as npt


class VidClass:
    """Handles video processing"""
    def __init__(self, path: str, scale: float) -> None:
        """ Initializes the VidClass

        :param path: directory that the program will be operating in
            this should be a directory with a single video in it, which both have the same name
        """
        # Initial Logger Settings
        fmt_main = "%(asctime)s | main\t\t: %(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%D %H:%M:%S")
        self.path = path
        self.scale = scale
        self.name = os.path.basename(self.path)
        self.frame_arr = self.get_vid()

    def get_vid(self) -> list:
        """Puts all frames into an array

        :returns: frames - list containing each frame from the '.mp4' file
        """
        logging.info("Reading video...")
        logging.info("This takes a minute or two...")
        frames = []
        cap = cv2.VideoCapture(self.path + "/" + self.name + ".mp4")

        # Check if camera opened successfully
        if cap.isOpened() is False:
            logging.info("Error opening video stream or file")

        # Read until video is completed
        while cap.isOpened():
            ret, frame = cap.read()     # Capture frame-by-frame
            if ret is True:
                if self.scale != 1:
                    frame = self.resize_img(frame, self.scale)

                frames.append(frame)
            else:
                break

        logging.info("Video length = %d frames", len(frames))

        cap.release()
        return frames

    def play_vid(self) -> None:
        """Plays the frames on loop in order

        Plays at about 30 frames per second by default
        Press 'q' to exit
        """
        logging.info("Playing video...")
        loop = True

        while loop:
            for mrk, frame in enumerate(self.frame_arr):
                # logging.info("Frame %d, %d.%d seconds", mrk, int(mrk/30), mrk%30)
                cv2.imshow('Frame',frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(3) & 0xFF == ord('q'):
                    loop = True
                    break

        cv2.destroyAllWindows()

    def select_frames(self, start=0, end=1, interval=1) -> None:
        """Outputs a selection of frames to a subdirectory './1_orig_frames'

        :param start: initial frame for selection
        :param end: final frame in selection
        :param interval: amount of frames to increment per output image
        """
        mrk = start
        while mrk <= end:
            cv2.imwrite(f"{self.path}/1_orig_frames/frame{mrk}.jpg", self.frame_arr[mrk])
            mrk += interval

    def resize_img(self, img: npt.NDArray[Any], scale: float) -> npt.NDArray[Any]:
        """Scales the image by a given ratio

        :param img: numpy image - original to be modified
        :param scale: float - the percentage by which the image will be expanded or compressed
        :returns: scaled_img - the modified image
        """
        w_1 = img.shape[1]
        h_1 = img.shape[0]

        w_2 = int(w_1 * scale)
        h_2 = int(h_1 * scale)

        new_dim = (w_2, h_2)

        scaled_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        return scaled_img
