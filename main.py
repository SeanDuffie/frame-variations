"""
main file
"""
import logging
import os
import sys
from tkinter import filedialog

from generate_variants import ImgMod

def main():
    # Initial Logger Settings
    fmt_main = "%(asctime)s | main\t\t: %(message)s"
    logging.basicConfig(format=fmt_main, level=logging.INFO,
                    datefmt="%Y-%m-%D %H:%M:%S")

    # If variants doesn't exist then create it
    if not os.path.exists("./variants"):
        os.mkdir("./variants")
        logging.error("Error: the './variants' dir did not exist, it has now been created.")
        logging.error("Error: Add your videos there and run again")
        sys.exit(1)

    # Make sure the user added elements to the variants directory
    if len(os.listdir("./variants")) < 2:
        logging.error("Error: the './variants' dir did not exist, it has now been created.")
        logging.error("Error: Add your videos there and run again")
        sys.exit(1)

    # Move each video into its own directory
    for file_name in os.listdir("./variants"):
        rel_path = "./variants/" + file_name
        if file_name == "Readme.txt":
            continue
        if os.path.isdir(rel_path):
            continue
        if file_name.endswith(".mp4"):
            new_dir = file_name[0:len(file_name)-4]
            if not os.path.exists("./variants/" + new_dir):
                os.mkdir("./variants/" + new_dir)
            os.replace(rel_path, "./variants/" + new_dir + "/" + file_name)
        else:
            logging.warning("Warning: Extra file in variants directory")
    
    # Run modifications on each folder
    for file_name in os.listdir("./variants"):
        rel_path = "./variants/" + file_name
        if file_name == "Readme.txt":
            continue
        elif os.path.isdir(rel_path):
            im = ImgMod(path=rel_path)
            im.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
