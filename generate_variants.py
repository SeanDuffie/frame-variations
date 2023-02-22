"""
Python Script intended to apply a selection of modifications to a selected image
generating many variations of the same image to see what effect is had on facial recognition

This will be done in batch on a large selection of files
TODO: Should the program clear the output directories before running?
TODO: Graypoint calculation
TODO: Merge white_balance and hue_sat into an hsv function
"""
import logging
import os
import sys
from tkinter import filedialog
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from frame_extractor import VidClass

### FLAGS ###
FACE = False    # Should it search for faces?
PREV = False     # Display output to screen
OUT = True      # Write output to a jpg file

# NOTE: If the resolution is too high and not downscaled, the program will run slowly
RESIZE = .25      # Factor to resize frames by? (1 skips calculation, must be greater than 0)


class ImgMod:
    """Applied image modifications automatically in bulk"""
    def __init__(self) -> None:
        # Initial Logger Settings
        fmt_main = "%(asctime)s | main\t\t: %(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%D %H:%M:%S")

        self.path = filedialog.askdirectory()
        if not os.path.exists(self.path + "/frames"):
            os.mkdir(self.path + "/frames")

        # Source Image Array
        self.img_arr = {}

    def acquire_frames(self, s_f=-1, e_f=-1, i=-1) -> None:
        """Pick out frames from a video

        This file will preview the video on loop (if requested) to the user
        Then prompt them to select what range of frames should be output
        """
        video = VidClass(path=self.path, scale=RESIZE)

        if PREV:
            logging.info("Previewing Video frames...")
            video.play_vid()

        # print(len(self.img_arr))
        # Prompt the user for their frame selection
        while s_f < 0:# or s_f > len(self.img_arr):
            try:
                s_f = int(input("Start Frame: "))
            except ValueError:
                logging.info("enter an int!")
                s_f = -1
        while e_f < s_f:# or e_f > len(self.img_arr):
            try:
                e_f = int(input("End Frame: "))
            except ValueError:
                logging.info("enter an int!")
                e_f = -1
        while i < 1:
            try:
                i = int(input("Interval between Frames: "))
            except ValueError:
                logging.info("enter an int!")
                i = -1

        # Write the frames to the frames directory
        video.select_frames(s_f, e_f, i)

        cv2.destroyAllWindows()

    def parse_dir(self) -> None:
        """Parses a directory of frames and reads in images to a list"""
        frame_dir = self.path + "/frames/"

        for file_name in os.listdir(frame_dir):
            # check if the image ends with png or jpg or jpeg
            if (file_name.endswith(".png") or file_name.endswith(".jpg")\
                or file_name.endswith(".jpeg")):
                logging.info("Reading File: %s...", file_name)
                img = cv2.imread(frame_dir + file_name)

                if img is None:
                    logging.info('Could not open or find the image: %s', file_name)
                    exit(0)

                self.img_arr[file_name] = img

        return 0

    def find_faces(self, img: npt.NDArray[Any]) -> list:
        """Scans an image for faces

        :param img: input image to be scanned
        :returns: faces - an array of tuples (x, y, width, height) where a face was detected
        """
        # Convert to Greyscale
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gry,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        return faces

    def auto_balance(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Automatically acquires the brightest and darkest points on the face

        Currently this is accomplished by locating the brightest and darkest points in the image,
        and adjusting the black and white points accordingly.

        NOTE: For best results, ensure that the input is cropped as tightly as possible around face

        :param img: input image, this will usually already be cropped around an identified face
        :returns: b_p - the automatic blackpoint
        :returns: w_p - the automatic whitepoint
        :returns: bal_img - the modified image
        """
        # Convert to HSV and split
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        # Identify the Black and White points automatically
        bp = np.min(v)
        wp = np.max(v)

        # Produce an adjusted image
        bal_img = self.white_balance(cur_img=img, b_p=bp, w_p=wp)

        cv2.imshow("fc", bal_img)                # Show auto balanced

        return bal_img

    def white_balance(self, cur_img: npt.NDArray[Any], b_p=0, w_p=0, g_p=0) -> npt.NDArray[Any]:
        """Adjust the white balance of the image

        Accomplished by modifying the blackpoint, whitepoint, and greypoint.

        :param cur_img: input BGR pixel array
        :param b_p: black point - this value and anything below it will be set to zero
        :param w_p: white point - this value and anything below it will be set to 255
        :param g_p: gray point - *NOT YET IMPLEMENTED*

        TODO: Add graypoint into calculation
        """
        # Convert to HSV and split
        hsv_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_img)

        # Modify value channel by clipping, subtracting, and scaling
        scale = (w_p-b_p)/255
        v_new = np.clip(v, b_p, w_p) - b_p         # Clipped/Subtracted Values
        v_new = (v_new / scale).astype(np.uint8)      # Scaled Values

        # logging.info("\tOriginal = (%d, %d)", np.min(v), np.max(v))
        
        # Recombine channels
        hsv_new = cv2.merge([h,s,v_new])

        # Convert back to bgr
        n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        return n_img

    def hue_sat(self, cur_img: npt.NDArray[Any], hmod=0, smod=0) -> npt.NDArray[Any]:
        """Modifies the Hue and Saturation of an Image

        Hue will loop if it overflows past the maximum or minimum
        Saturation will cap if it reaches a maximum or minimum value

        :param cur_img: input BGR pixel array
        :param hmod: hue offset - all hue values will be *incremented* by this number
        :param smod: saturation percentage - all saturation values will be *scaled* by this percentage
        REMOVE: :param vmod: value offset - all values will be *incremented* by this number
        """
        # Convert to HSV and split
        hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        # Modify channels by adding difference and modulo 180
        h_new = np.mod(h + hmod, 180).astype(np.uint8)
        s_new = s*(1+(smod-1)/100)
        s_new = np.clip(s_new.astype(np.uint8), 0, 255)

        # Recombine channels
        hsv_new = cv2.merge([h_new,s_new,v])

        # Convert back to bgr
        n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        return n_img

    def color_mod(self, cur_img: npt.NDArray[Any], b_scale: float,
                    g_scale: float, r_scale: float) -> npt.NDArray[Any]:
        """Adjusts the Intensity of each BGR value individually

        :param cur_img: input BGR pixel array
        :param b_scale: percentage value to scale blue pixels
        :param g_scale: percentage value to scale green pixels
        :param r_scale: percentage value to scale red pixels
        """
        b,g,r = cv2.split(cur_img)

        b = b * b_scale
        g = g * g_scale
        r = r * r_scale

        b = np.clip(b, 0, 255)
        g = np.clip(g, 0, 255)
        r = np.clip(r, 0, 255)

        n_img = cv2.merge([b,g,r])

        return n_img

    def write_file(self, mod: str, name: str, img: npt.NDArray[Any]) -> None:
        """Writes the input image to a file

        This is necessary to organize the output files into different directories
        If the directory does not currently exist, it will be created

        :param mod: str - name of the modification, it will have a its own directory
        :param name: str - name of the input image, it will also be the output name (e.g frame27)
        :param img: modified image that will be saved
        """
        if not os.path.exists(self.path + mod):
            os.mkdir(self.path + mod)
        cv2.imwrite(self.path + mod + name, img)

    def run(self) -> None:
        """Main Class Runner

        This is where most of the actual process will occur
        If any changes are needed by the end user, they will likely be done here
        """
        ##########   START LOAD IMAGES   ##########
        self.acquire_frames()
        self.parse_dir()

        for file_name, cur_img in self.img_arr.items():
            if FACE:
                ##########   START FACES   ##########
                logging.info("Finding faces...")
                faces = self.find_faces(img=cur_img)

                if len(faces) == 0:
                    logging.info("No faces!")
                    continue

                logging.info("%d faces found!", len(faces))
                # Draw a rectangle around the faces
                cropped = cur_img
                for (x, y, w, h) in faces:
                    cropped = cur_img[y:y+h, x:x+w]
                    cv2.imshow("cropped", cropped)
                    logging.info("Press 'c' to confirm. Any other button skips.")
                    if cv2.waitKey(0) & 0xFF == ord('c'):
                        # logging.info(x, w, y, h)
                        auto = self.auto_balance(cropped)         # Auto
                        cv2.rectangle(cur_img, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        self.write_file("/auto/", file_name, auto)
                ########   END FACES   ########

            cv2.imshow("original", cur_img)     # Show Original
            #########    END LOAD IMAGES    #########


            # NOTE: IF ANY ADJUSTMENTS ARE NEEDED, THEY WILL LIKELY BE IN THE 2-3 FOLLOWING SECTIONS
            ########## ADJUSTMENTS ##########
            # Brightness/Contrast using Black and White points
            fc_1 = self.white_balance(cur_img, 0, 190)
            fc_2 = self.white_balance(fc_1, 0, 200)
            fc_3 = self.white_balance(fc_2, 50, 200)
            fc_12_220 = self.white_balance(cur_img, 12, 220)
            fc_24_185 = self.white_balance(cur_img, 24, 185)

            # Hue/Saturation
            low_sat_45 = self.hue_sat(fc_24_185, 45, 1)
            high_sat_45 = self.hue_sat(fc_24_185, 45, 100)
            low_sat_90 = self.hue_sat(fc_24_185, 90, 1)
            high_sat_90 = self.hue_sat(fc_24_185, 90, 100)
            low_sat_135 = self.hue_sat(fc_24_185, 135, 1)
            high_sat_135 = self.hue_sat(fc_24_185, 135, 100)
            low_sat_180 = self.hue_sat(fc_24_185, 180, 1)
            high_sat_180 = self.hue_sat(fc_24_185, 180, 100)
            ########   END ADJUSTMENTS   ##########


            ##########  START DISPLAY  ##########
            if PREV:
                # 0 Hue
                cv2.imshow("low_sat", low_sat_180)
                cv2.imshow("high_sat", high_sat_180)
                # cv2.waitKey(0)
                
                # 60 Hue
                cv2.imshow("low_sat", low_sat_45)
                cv2.imshow("high_sat", high_sat_45)
                # cv2.waitKey(0)
                
                # 120 Hue
                cv2.imshow("low_sat", low_sat_90)
                cv2.imshow("high_sat", high_sat_90)
                # cv2.waitKey(0)
                
                # 150 Hue
                cv2.imshow("low_sat", low_sat_135)
                cv2.imshow("high_sat", high_sat_135)
                cv2.waitKey(0)

                # Raw BGR Color Modification
                # cv2.imshow("color_mod", color_mod(cur_img, 0.9, 0.9, 0.9))
                # cv2.imshow("blue", color_mod(cur_img, 1, 0, 0))
                # cv2.imshow("green", color_mod(cur_img, 0, 1, 0))
                # cv2.imshow("red", color_mod(cur_img, 0, 0, 1))
        
                cv2.destroyAllWindows()
            ##########   END DISPLAY   ##########


            ##########  START FILE OUTPUT  ##########
            if OUT:
                self.write_file("/g1Level_0-190/", file_name, fc_1)
                self.write_file("/g2Level_0-190_0-200/", file_name, fc_2)
                self.write_file("/g3Level_0-190_0-200_50-200/", file_name, fc_3)
                self.write_file("/Level_12-220/", file_name, fc_12_220)
                self.write_file("/Level_24-185/", file_name, fc_24_185)

                self.write_file("/level_24-1-185_h_90/", file_name, low_sat_45)
                self.write_file("/level_24-1-185_h_-90/", file_name, low_sat_135)
                self.write_file("/level_24-1-185_h_90_s_100/", file_name, high_sat_45)
                self.write_file("/level_24-1-185_h_-90_s100/", file_name, high_sat_135)

                self.write_file("/level_24-1-185_h_-180/", file_name, low_sat_90)
                self.write_file("/level_24-1-185_h_-180_s_100/", file_name, high_sat_90)

                self.write_file("/level_24-1-185_s_100/", file_name, high_sat_180)
                self.write_file("/level_24-1-185_s_-0/", file_name, low_sat_180)
            ##########   END FILE OUTPUT   ##########


    # def pixel_print(self, name: str, cur_img: npt.NDArray[Any]) -> None:
    #     """
    #     Debugging Pixel Values
    #     """
    #     hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
    #     b,g,r = cv2.split(cur_img)
    #     h,s,v = cv2.split(hsv)

    #     logging.info("%s: ", name)
    #     logging.info("\tOriginal = (%d, %d)", np.min(v), np.max(v))


if __name__ == "__main__":
    im = ImgMod()
    sys.exit(im.run())
