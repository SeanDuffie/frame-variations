"""
Python Script intended to apply a selection of modifications to a selected image
generating many variations of the same image to see what effect is had on facial recognition

This will be done in batch on a large selection of files
TODO: Graypoint calculation
TODO: Adjust the auto balance by Stoian's suggestion (max of min/max of max)
FIXME: Naming convention for frames cause 90 to be seen as higher than 100, do 90 -> 090 instead
"""
import logging
import os
import sys
from tkinter import filedialog

import cv2
import numpy as np

from frame_extractor import VidClass

### FLAGS ###
FACE = False            # Should it search for faces?
PREV = True            # Display output to screen
OUT = True             # Write output to a jpg file
FRESH = True           # Removes all existing generated frames
VIDEO = True           # Set false to skip reading in the video, runtime is much faster without
MANUAL_FRAMES = False   # Set true if you want to manually pick the start, end, and interval

# NOTE: If the resolution is too high and not downscaled, the program will run slowly
RESIZE = .25      # Factor to resize frames by? (1 skips calculation, must be greater than 0)


class ImgMod:
    """Applies image modifications automatically in bulk"""
    def __init__(self, path="") -> None:
        # Initial Logger Settings
        fmt_main = "%(asctime)s | ImgMod:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%D %H:%M:%S")

        self.path = path
        if self.path == "":
            self.path = filedialog.askdirectory()
        if self.path == "":
            logging.error("Error: No path specified! Exiting...")
            sys.exit(1)
        self.clean_setup()

        # Source Image Array
        self.img_arr = {}

    def clean_setup(self) -> None:
        """Sets up a fresh environment
        Relies on the "FRESH" and "VIDEO" flags

        FRESH decides whether the environment should have all old output files deleted
        VIDEO decides whether the old frames previously captured from the video should be deleted
            and recaptured

        All that is needed for this setup is to have a directory with a video contained within
            that has the same name, everything else will be auto populated
        """
        vid_found = 0
        if not VIDEO:
            logging.info("Using old frames! If this is a new directory then please set global 'VIDEO' to true")

        for file_name in os.listdir(self.path):
            f = os.path.join(self.path, file_name)

            # Locate the source video file
            if os.path.isfile(f) and ".mp4" in file_name:
                logging.info("Video found at: %s", f)
                vid_found += 1

            else:
                if FRESH:
                    # Locate original frame directory and either clear or reset it
                    if VIDEO or file_name != "1_orig_frames":
                        logging.info("Removing Directory: %s", f)
                        for sub_dir_elem in os.listdir(f):
                            os.remove(f + "/" + sub_dir_elem)
                        os.rmdir(f)

        if not os.path.exists(self.path + "/1_orig_frames"):
            os.mkdir(self.path + "/1_orig_frames")
        if not os.path.exists(self.path + "/2_grayscale"):
            os.mkdir(self.path + "/2_grayscale")
        if not os.path.exists(self.path + "/3_auto"):
            os.mkdir(self.path + "/3_auto")

        if vid_found == 0:
            logging.error("Error: No videos found")
            sys.exit(1)
        elif vid_found > 1:
            logging.error("Error: Multiple videos found")
            sys.exit(1)

    def acquire_frames(self, s_f=-1, e_f=-1, i=-1) -> None:
        """Pick out frames from a video

        This file will preview the video on loop (if requested) to the user
        Then prompt them to select what range of frames should be output
        """
        video = VidClass(path=self.path, scale=RESIZE)

        if PREV:
            logging.info("Previewing Video frames...")
            video.play_vid()

        ## DEBUG
        s_f = 0
        e_f = len(video.frame_arr) - 1
        i = 10


        if MANUAL_FRAMES:
            e_f = 0
            # Prompt the user for their frame selection
            while s_f < 0 or s_f >= len(video.frame_arr) - 1:
                try:
                    s_f = int(input("Start Frame: "))
                except ValueError:
                    logging.warning("enter an int!")
                    s_f = -1
            while e_f < s_f or e_f >= len(video.frame_arr):
                try:
                    e_f = int(input("End Frame: "))
                except ValueError:
                    logging.warning("enter an int!")
                    e_f = -1
            while i < 1:
                try:
                    i = int(input("Interval between Frames: "))
                except ValueError:
                    logging.warning("enter an int!")
                    i = -1

        # Write the frames to the frames directory
        video.select_frames(s_f, e_f, i)


    def parse_dir(self) -> None:
        """Parses a directory of frames and reads in images to a list"""
        frame_dir = self.path + "/1_orig_frames/"

        for file_name in os.listdir(frame_dir):
            # check if the image ends with png or jpg or jpeg
            if (file_name.endswith(".png") or file_name.endswith(".jpg")
                or file_name.endswith(".jpeg")):
                logging.info("Reading File: %s...", file_name)
                img = cv2.imread(frame_dir + file_name)

                if img is None:
                    logging.info('Could not open or find the image: %s', file_name)
                    sys.exit(1)

                self.img_arr[file_name] = img

    def find_faces(self, gry_img) -> list:
        """Scans an image for faces

        :param img: input image to be scanned
        :returns: faces - an array of tuples (x, y, width, height) where a face was detected
        """
        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gry_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        return faces

    def auto_balance(self, img) -> tuple:
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
        b_p = np.min(v)
        w_p = np.max(v)

        return b_p, w_p

    def adjust(self, cur_img, fname: str, mod_name = "", color=(0, 1), balance = (0, 255)):
        """Adjust the white balance and hue/saturation of the image at the same time

        Accomplished by modifying the blackpoint, whitepoint, and graypoint.

        :param cur_img: input BGR pixel array
        :param fname: str - name of the input image, it will also be the output name (e.g frame27)
        :param color: tuple input that contains the hue and saturation inputs
            :param hmod: hue offset - all hue values will be *incremented* by this number
            :param smod: saturation percentage - saturation values will be *scaled* by this percent
        :param balance: tuple input that contains the balance inputs
            :param b_p: black point - this value and anything below it will be set to zero
            :param w_p: white point - this value and anything below it will be set to 255
            :param g_p: gray point - *NOT YET IMPLEMENTED*

        TODO: Add graypoint into calculation
        """
        # Read in tuple input parameters
        h_mod, s_mod = color
        b_p, w_p = balance
        if mod_name == "":
            mod_name = f"/h{h_mod}_s{s_mod}_b{b_p}_w{w_p}/"
        # Convert to HSV and split
        hsv_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_img)

        ##########   START ADJUSTMENTS   ##########
        # Modify Hue by applying a mod
        h_new = np.mod(h + h_mod, 180).astype(np.uint8)

        # Modify Saturation clipping, subtracting, and scaling
        s_new = s*(1+(s_mod-1)/100)
        s_new = np.clip(s_new.astype(np.uint8), 0, 255)

        # Modify Value (White Balance) by clipping, subtracting, and scaling
        scale = (w_p-b_p)/255
        v_new = np.clip(v, b_p, w_p) - b_p         # Clipped/Subtracted Values
        v_new = (v_new / scale).astype(np.uint8)      # Scaled Values

        # Recombine channels
        hsv_new = cv2.merge([h_new,s_new,v_new])

        # Convert back to bgr
        n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        ##########    END ADJUSTMENTS    ##########


        ##########   START PREVIEW   ##########
        if PREV:
            cv2.imshow(mod_name, n_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        ##########    END PREVIEW    ##########


        ##########   START OUTPUT   ##########
        if OUT:
            if not os.path.exists(self.path + mod_name):
                os.mkdir(self.path + mod_name)
            cv2.imwrite(self.path + mod_name + fname, n_img)
        ##########    END OUTPUT    ##########

        return n_img


    def run(self) -> None:
        """Main Class Runner

        This is where most of the actual process will occur
        If any changes are needed by the end user, they will likely be done here
        """
        ##########   START LOAD IMAGES   ##########
        if VIDEO:
            self.acquire_frames()           # Grabs the frames from the video, if enabled
        self.parse_dir()                    # Grabs the generated frames from "./1_orig_frames"
        ##########    END LOAD IMAGES    ##########

        ##########   START IMAGE MODS   ##########
        for file_name, cur_img in self.img_arr.items():
            # Initial Grayscale
            gry_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(self.path + "/2_grayscale/" + file_name, gry_img)

            ##########   START FACES   ##########
            if FACE:
                logging.info("Finding faces...")
                faces = self.find_faces(gry_img=gry_img)

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
                        a_bp, a_wp = self.auto_balance(cropped)         # Auto
                        self.adjust(cur_img=cur_img, fname=file_name, mod_name="/3_auto/", balance=(a_bp, a_wp))
            ########   END FACES   ########

            ########## ADJUSTMENTS ##########
            # NOTE: IF ANY ADJUSTMENTS ARE NEEDED, THEY WILL BE MADE HERE
            # Brightness/Contrast using Black and White points
            fc_1 = self.adjust(cur_img=cur_img, fname=file_name, balance=(0, 190))
            fc_2 = self.adjust(cur_img=fc_1, fname=file_name, balance=(0, 200))
            self.adjust(cur_img=fc_2, fname=file_name, balance=(50, 200))
            self.adjust(cur_img=cur_img, fname=file_name, balance=(12, 220))
            self.adjust(cur_img=cur_img, fname=file_name, balance=(24, 185))

            # Hue/Saturation
            self.adjust(cur_img=cur_img, fname=file_name, color=(45,1), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(45,100), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(90,1), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(90,100), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(135,1), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(135,100), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(180,1), balance=(24, 185))
            self.adjust(cur_img=cur_img, fname=file_name, color=(180,100), balance=(24, 185))
            ########   END ADJUSTMENTS   ##########
        ##########    END IMAGE MODS    ##########


    # def color_mod(self, cur_img, b_scale: float, g_scale: float, r_scale: float):
    #     """Adjusts the Intensity of each BGR value individually

    #     :param cur_img: input BGR pixel array
    #     :param b_scale: percentage value to scale blue pixels
    #     :param g_scale: percentage value to scale green pixels
    #     :param r_scale: percentage value to scale red pixels
    #     """
    #     b,g,r = cv2.split(cur_img)

    #     b = b * b_scale
    #     g = g * g_scale
    #     r = r * r_scale

    #     b = np.clip(b, 0, 255)
    #     g = np.clip(g, 0, 255)
    #     r = np.clip(r, 0, 255)

    #     n_img = cv2.merge([b,g,r])

    #     return n_img


if __name__ == "__main__":
    im = ImgMod()
    sys.exit(im.run())
