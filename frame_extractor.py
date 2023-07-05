""" vid_test.py
"""
import logging
import os
import sys
import time
from typing import Any

import cv2
import keyboard
import numpy.typing as npt


class FrameExt:
    """Handles video processing"""
    def __init__(self, path: str, scale: float = .25) -> None:
        """ Initializes the FrameExt

        :param path: directory that the program will be operating in
            this should be a directory with a single video in it, which both have the same name
        """
        # Initial Logger Settings
        fmt_main: str = "%(asctime)s | %(levelname)s |\tFrameExt:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%D %H:%M:%S")
        self.path: str = path
        self.scale: float = scale
        self.name: str = os.path.basename(self.path)
        self.cap = cv2.VideoCapture(self.path + "/" + self.name + ".mp4")
        self.width, self.height, self.fcnt, self.fps = {
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            self.cap.get(cv2.CAP_PROP_FPS),
        }
        self.index: int = 0

    def get_frame(self, index: int = -1) -> npt.NDArray[Any]:
        # Check for video bounds
        if index >= self.fcnt:
            logging.error("Index out of bounds")
            sys.exit(1)
        if index == -1:
            index = self.index

        # Set the index and read next value
        if self.index == index:
            ret, image = self.cap.read()
            self.index += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, image = self.cap.read()
            self.index = index + 1

        # Check if the frame was acquired successfully
        if ret:
            if self.scale != 1:
                frame = self.resize_img(image, self.scale)
            return image
        else:
            logging.error("ret failed")
            sys.exit(1)

    def play_vid(self) -> None:
        """Plays the frames on loop in order

        Plays at about 30 frames per second by default
        Press 'q' to exit
        """
        logging.info("Playing video...")
        temp_cap = cv2.VideoCapture(self.path + "/" + self.name + ".mp4")

        while temp_cap.isOpen():
            ret, frame = temp_cap.read()

            if ret:
                cv2.imshow('Frame',frame)
                time.sleep(1/self.fps)
            else:
                break

            # Press Q on keyboard to  exit
            if keyboard.is_pressed("q"):
                break

        cv2.destroyAllWindows()

    def resize_img(self, img, scale: float):
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
