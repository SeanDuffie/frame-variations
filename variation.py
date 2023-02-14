"""
Python Script intended to apply a selection of modifications to a selected image
generating many variations of the same image to see what effect is had on facial recognition

This will be done in batch on a large selection of files
"""
import os
import sys
from tkinter import Button, Label, Tk, filedialog
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

### FLAGS ###
VIZ = True     # Display output to screen
OUTPUT = True   # Write output to a jpg file
VIDEO = True   # True means to populate ./raw with frames from a video, False uses ./raw as is


# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# Source Image Array
IMG_ARR = {}

# def resize_img(img, scale):
#     """
#     Scales the image by the ratio passed in scale
#     """
#     w1 = img.shape[1]
#     h1 = img.shape[0]
#     w2 = int(w1 * scale)
#     h2 = int(h1 * scale)
#     new_dim = (w2, h2)
#     return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

def acquire_frames():
    """
    Pick out frames from a video
    """
    # open a file chooser dialog and allow the user to select an input image
    path = filedialog.askopenfilename()
    print("Path:", path)

def parse_dir():
    """ Parses a directory of images and calls the function to modify images """
    for file_name in os.listdir("./raw"):
        # check if the image ends with png or jpg or jpeg
        if (file_name.endswith(".png") or file_name.endswith(".jpg")\
            or file_name.endswith(".jpeg")):
            print("Reading File: " + file_name + "...")
            img = cv2.imread("raw/" + file_name)

            if img is None:
                print('Could not open or find the image: img/' + file_name)
                exit(0)

            IMG_ARR[file_name] = img

    return 0

def bri_con(cur_img, b, w) -> npt.NDArray[Any]:
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
    
    # Recombine channels
    hsv_new = cv2.merge([h,s,s_v.astype(np.uint8)])

    # Convert back to bgr
    n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    return n_img

def hue_sat(cur_img, hmod, smod, vmod=0) -> npt.NDArray[Any]:
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

    print("Saturation:", hsv[:,:,1].mean())

    # Modify channels by adding difference and modulo 180
    hnew = np.mod(h + hmod, 180).astype(np.uint8)
    snew = s*(1+(smod)/100)
    snew = np.clip(snew.astype(np.uint8), 0, 255)
    vnew = np.mod(v + vmod, 256).astype(np.uint8)
    
    # Recombine channels
    hsv_new = cv2.merge([hnew,snew,vnew])
    
    print("Saturation:", hsv_new[:,:,1].mean())

    # Convert back to bgr
    n_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    return n_img


def color_mod(cur_img, b_scale, g_scale, r_scale) -> npt.NDArray[Any]:
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

# def mod_brightness_contrast(img, a, b):
#     """
#     https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
#     Brightness and contrast adjustments
#         Two commonly used point processes are multiplication and addition with a constant:

#         g(x)=αf(x)+β
#         The parameters α>0 and β are often called the gain and bias parameters; sometimes these parameters are said to control contrast and brightness respectively.
#         You can think of f(x) as the source image pixels and g(x) as the output image pixels. Then, more conveniently we can write the expression as:

#         g(i,j)=α⋅f(i,j)+β
#         where i and j indicates that the pixel is located in the i-th row and j-th column.

#         :param img: source image
#         :param a: (float) alpha value, often called gain or contrast. [1.0-3.0]
#         :param b: (int) beta value, often called bias or brightness. [0-100]
#     """
#     return cv2.convertScaleAbs(img, alpha=a, beta=b)

# def gamma_trans(img, gamma):
#     gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
#     gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
#     return cv2.LUT(img,gamma_table)

def pixel_print(name: str, cur_img) -> None:
    """
    Debugging Pixel Values
    """
    hsv = cv2.cvtColor(cur_img, cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(cur_img)
    h,s,v = cv2.split(hsv)

    print(name + ":")
    print("\tMax =", np.max(v))
    print("\tMin =", np.min(v))
    # row = ""
    # for y in b:
    #     row += str(y[40]) + " "
    # print(row)
    # print()

def main():
    """
    main
    """

    if VIDEO:
        acquire_frames()

    parse_dir()

    for file_name, cur_img in IMG_ARR.items():

        ### Greyscale
        gry = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)


        ##########   FACES   ##########
        # # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gry,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE #flags =cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
        # # Draw a rectangle around the faces
        # cropped = cur_img
        # for (x, y, w, h) in faces:
        #     print(x, w, y, h)
        #     cropped = cur_img[y:y+h, x:x+w]
        #     cv2.rectangle(cur_img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        ########   END FACES   ########
        x, w, y, h = 161, 143, 62, 143
        cropped = cur_img[y:y+h, x:x+w]



        ########## ADJUSTMENTS ##########
        # Brightness/Contrast using Black and White points
        # TODO: Add grey point
        fc_1 = bri_con(cur_img, 0, 190)
        fc_2 = bri_con(fc_1, 0, 200)
        fc_3 = bri_con(fc_2, 50, 200)
        fc_12_220 = bri_con(cur_img, 12, 220)
        fc_24_185 = bri_con(cur_img, 24, 185)

        pixel_print("cur_img", cur_img)
        pixel_print("cropped", cropped)
        pixel_print("fc_24_185", fc_24_185)

        # Hue/Saturation
        low_sat_45 = hue_sat(fc_24_185, 45, 1)
        high_sat_45 = hue_sat(fc_24_185, 45, 100)
        low_sat_90 = hue_sat(fc_24_185, 90, 1)
        high_sat_90 = hue_sat(fc_24_185, 90, 100)
        low_sat_135 = hue_sat(fc_24_185, 135, 1)
        high_sat_135 = hue_sat(fc_24_185, 135, 100)
        low_sat_180 = hue_sat(fc_24_185, 180, 1)
        high_sat_180 = hue_sat(fc_24_185, 180, 100)
        ########   END ADJUSTMENTS   ##########


        ##########  START FILE OUTPUT  ##########
        if OUTPUT:
            cv2.imwrite("edited/frame_" + file_name, cur_img)
            # cv2.imwrite("edited/2_gray_" + file_name, gry)
            # cv2.imwrite("edited/cropped" + file_name, cropped)


            cv2.imwrite("edited/g1Level_0-190_" + file_name, fc_1)
            cv2.imwrite("edited/g2Level_0-190_0-200_" + file_name, fc_2)
            cv2.imwrite("edited/g3Level_0-190_0-200_50-200_" + file_name, fc_3)
            cv2.imwrite("edited/Level_12-220_" + file_name, fc_12_220)
            cv2.imwrite("edited/Level_24-185_" + file_name, fc_24_185)


            cv2.imwrite("edited/level_24-1-185_h_90" + file_name, low_sat_45)
            cv2.imwrite("edited/level_24-1-185_h_-90" + file_name, low_sat_135)
            cv2.imwrite("edited/level_24-1-185_h_90_s_100" + file_name, high_sat_45)
            cv2.imwrite("edited/level_24-1-185_h_-90_s100" + file_name, high_sat_135)

            cv2.imwrite("edited/level_24-1-185_h_-180" + file_name, low_sat_90)
            cv2.imwrite("edited/level_24-1-185_h_-180_s_100" + file_name, high_sat_90)

            cv2.imwrite("edited/level_24-1-185_s_100" + file_name, high_sat_180)
            cv2.imwrite("edited/level_24-1-185_s_-0" + file_name, low_sat_180)
        ##########   END FILE OUTPUT   ##########

        ##########  START DISPLAY  ##########
        if VIZ:
            cv2.imshow("original", cur_img)     # Show Original
            cv2.imshow("greyscale", gry)        # Show Greyscale
            cv2.imshow("cropped", cropped)      # Show Isolated Face

            # 0 Hue
            cv2.imshow("low_sat", low_sat_180)
            cv2.imshow("high_sat", high_sat_180)
            cv2.waitKey(0)
            
            # 60 Hue
            cv2.imshow("low_sat", low_sat_45)
            cv2.imshow("high_sat", high_sat_45)
            cv2.waitKey(0)
            
            # 120 Hue
            cv2.imshow("low_sat", low_sat_90)
            cv2.imshow("high_sat", high_sat_90)
            cv2.waitKey(0)
            
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
    return 0

if __name__ == "__main__":
    sys.exit(main())
