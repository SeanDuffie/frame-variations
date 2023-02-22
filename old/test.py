import os
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd


def load_images(folder):
    print(folder)
    print(os.path.basename(folder))
    images = []
    for file in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,file))
        if image is not None:
            images.append(image)
            
    return images

if __name__ == "__main__":
    print("pick directory")
    path = filedialog.askdirectory()
    # images = load_images('C:/Users/sduffie/OneDrive - Texas A&M University/Work/ARM-208/Izaiah_work_2-15/frames/PXL_20221212_894/894_frames')
    images = load_images(path)
    print("done loading")

    bmin = []
    bmax = []
    gmin = []
    gmax = []
    rmin = []
    rmax = []
    max_chnl = []
    min_chnl = []

    for img in images:

        bmin.append(img[..., 0].min())  # blue channel
        bmax.append(img[..., 0].max())

        gmin.append(img[..., 1].min())  # green channel
        gmax.append(img[..., 1].max())

        rmin.append(img[..., 2].min()) # red channel
        rmax.append(img[..., 2].max())
        max_chnl.append(np.max([np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2])]))
        min_chnl.append(np.min([np.min(img[:,:,0]), np.min(img[:,:,1]), np.min(img[:,:,2])]))

    print(rmin)
    print(rmax)

    df = pd.DataFrame({'Blue Max': bmax,
                    'Blue Min': bmin,
                    'Green Max': gmax,
                    'Green Min': gmin,
                    'Red Max': rmax,
                    'Red Min': rmin,
                    'Overall Max': max_chnl,
                    'Overall Min': min_chnl})
    print(df)

    df.to_csv('min_max ' + 'vid' + '.csv', index=False)