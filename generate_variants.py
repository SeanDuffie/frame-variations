"""
Python Script intended to apply a selection of modifications to a selected image
generating many variations of the same image to see what effect is had on facial recognition

This will be done in batch on a large selection of files
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
PREV = True     # Display output to screen
OUT = True      # Write output to a jpg file


class ImgMod:
    """Applied image modifications automatically in bulk"""
    def __init__(self, prev, out):
        # Initial Logger Settings
        fmt_main = "%(asctime)s | main\t\t: %(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%D %H:%M:%S")

        self.prev = prev    # Display output to screen
        self.out = out      # Write output to a jpg file

        self.path = filedialog.askdirectory()
        if not os.path.exists(self.path + "/frames"):
            os.mkdir(self.path + "/frames")

        # Source Image Array
        self.img_arr = {}

    def acquire_frames(self) -> None:
        """
        Pick out frames from a video
        """
        video = VidClass(self.path)

        logging.info("Previewing Video frames...")
        video.play_vid()

        a = -1
        b = -1
        i = -1
        while a < 0:
            try:
                a = int(input("start frame: "))
            except:
                logging.info("enter an int!")
                a = -1
        while b < a:
            try:
                b = int(input("end frame: "))
            except:
                logging.info("enter an int!")
                b = -1
        while i < 1:
            try:
                i = int(input("interval: "))
            except:
                logging.info("enter an int!")
                i = -1
        video.select_frames(a, b, i)
        
        cv2.destroyAllWindows()

    def parse_dir(self) -> None:
        """ Parses a directory of images and calls the function to modify images """
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

    def find_faces(self, img) -> list:
        """ detect faces """
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")      # Create the haar cascade
        ##########   FACES   ##########
        # Detect faces in the image
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                     # Greyscale
        faces = faceCascade.detectMultiScale(
            gry,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE #flags =cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        ########   END FACES   ########
        return faces

    def auto_balance(self, img):
        """ automatically acquire the brightest and darkest point on the face """
        # Convert to HSV and split
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        bp = np.min(v)
        wp = np.max(v)

        bal_img = self.bri_con(cur_img=img, b=bp, w=wp)

        cv2.imshow("fc", bal_img)                # Show auto balanced

        return bal_img

    def bri_con(self, cur_img, b=0, w=0, g=0) -> npt.NDArray[Any]:
        """
        Adjust the brightness/contrast of the image by modifying the blackpoint, whitepoint, and greypoint

        :param cur_img: input BGR pixel array
        :param b: black point - this value and anything below it will be set to zero
        :param w: white point - this value and anything below it will be set to 255

        TODO: How do I change greypoint???
        """
        # Convert to HSV and split
        hsv_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_img)

        # Modify value channel by clipping at the black and white points
        # subtracting the black point
        # and scaling the whole thing so that the new white point is 255
        scale = (w-b)/255
        c_v = np.clip(v, b, w) - b              # Clipped Values
        s_v = c_v / scale                   # Scaled Values
        
        logging.info("\tOriginal = (%d, %d)", np.min(v), np.max(v))
        
        # Recombine channels
        hsv_new = cv2.merge([h,s,s_v.astype(np.uint8)])

        # Convert back to bgr
        n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        return n_img

    def hue_sat(self, cur_img, hmod=0, smod=0, vmod=0) -> npt.NDArray[Any]:
        """
        Modifies the Hue and Saturation of an Image

        :param cur_img: input BGR pixel array
        :param hmod: hue offset - all hue values will be incremented by this number (overflow loops)
        :param smod: saturation percentage - all saturation values will be **scaled** by this **percentage** (overflow loops)
        :param vmod: value offset - all values will be incremented by this number (overflow loops)
        """
        # Convert to HSV and split
        hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        # logging.info("Saturation:", hsv[:,:,1].mean())

        # Modify channels by adding difference and modulo 180
        hnew = np.mod(h + hmod, 180).astype(np.uint8)
        snew = s*(1+(smod-1)/100)
        snew = np.clip(snew.astype(np.uint8), 0, 255)
        vnew = np.mod(v + vmod, 256).astype(np.uint8)
        
        # Recombine channels
        hsv_new = cv2.merge([hnew,snew,vnew])

        # Convert back to bgr
        n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        return n_img

    def color_mod(self, cur_img, b_scale, g_scale, r_scale) -> npt.NDArray[Any]:
        """
        Adjusts the Intensity of each BGR value individually

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

    def write_file(self, mod, name, img) -> None:
        if not os.path.exists(self.path + mod):
            os.mkdir(self.path + mod)
        cv2.imwrite(self.path + mod + name, img)

    def run(self) -> None:
        """ Main Class Runner """
        self.acquire_frames()
        self.parse_dir()

        for file_name, cur_img in self.img_arr.items():
            bp, wp = 0, 255

            logging.info("Finding faces...")
            faces = self.find_faces(img=cur_img)
            if len(faces) == 0:
                logging.info("No faces!")
            else:
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


                cv2.imshow("original", cur_img)     # Show Original
                ########## ADJUSTMENTS ##########
                # Brightness/Contrast using Black and White points
                # TODO: Add grey point
                fc_1 = self.bri_con(cur_img, 0, 190)
                fc_2 = self.bri_con(fc_1, 0, 200)
                fc_3 = self.bri_con(fc_2, 50, 200)
                fc_12_220 = self.bri_con(cur_img, 12, 220)
                fc_24_185 = self.bri_con(cur_img, 24, 185)

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
                if self.prev:
                    # cv2.imshow("original", cur_img)     # Show Original
                    # cv2.imshow("greyscale", gry)        # Show Greyscale
                    # cv2.imshow("cropped", cropped)      # Show Isolated Face

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
                if self.out:
                    self.write_file("/edited/", file_name, cur_img)
                    # self.write_file("/2_gray_", file_name, gry)
                    # self.write_file("/cropped", file_name, cropped)

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

    def pixel_print(self, name: str, cur_img) -> None:
        """
        Debugging Pixel Values
        """
        hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
        b,g,r = cv2.split(cur_img)
        h,s,v = cv2.split(hsv)

        logging.info(name + ":")
        logging.info("\tOriginal = (%d, %d)", np.min(v), np.max(v))


if __name__ == "__main__":
    im = ImgMod(prev=PREV, out=OUT)
    sys.exit(im.run())