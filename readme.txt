# Instructions for the overlay software

1. Make sure python is installed
2. Open a command prompt or powershell
3. Navigate to the directory that "overlay.py" is contained in
4. Run `pip install -r requirements.txt`
5. Run `python overlay.py'
6. If BATCH is enabled, then you will be prompted to select a directory of videos,
else you will have to select an individual video
7. Depending on other settings, you may be prompted to adjust the brightness threshold
and the start/stop frames. Keybinds are listed below (must be pressed with image window in focus)
    - 'esc' - Skips the current video
    - 'enter' - accepts the current settings
    - 'space' - sets the current frame as the starting point
    - 'backspace' - sets the current frame as the ending point
    - 'left' - moves back one frame
    - 'right' - moves forward one frame
    - 'up' - increases the threshold
    - 'down' - decreases the threshold
8. Upon completion for each video, the output image will be saved to a folder named "./outputs"
on the local directory (if nothing is there then it will make one automatically)
    - If BATCH is enabled, then the batch of videos will receive their own folder inside the
    outputs directory with the same name as the original directory.

### Notes on Functionality

- In summary, the program will take all the photos from the video and overlay them on top of each
other. The goal of this is to show as many of the particles that passed through as possible in one
image, so that patterns can be recognizes on size and preferred path.

### Configuration options

- To run in batch, set the boolean "BATCH" at the top to True
    - This will run all the videos in a directory at once instead of one video at a time
    - Outputs will be placed in a nested directory inside "./outputs" with the same name as source
- To run with alpha blending, set ALPHA to True
    - ** This doesn't work as we planned, I only kept it for reference
- To manually adjust the Initial threshold value, or the default start and stop frame, use:
    - INIT_THRESH [default=255]
    - START [default=0]
    - STOP [default=-1]