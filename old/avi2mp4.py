from tkinter import filedialog
import moviepy.editor as moviepy

video = filedialog.askopenfilename()
if video == "":
    print("Error: no video selected")
    exit(1)
elif not video.endswith(".avi"):
    print("Error: input is not a .avi type")
    exit(2)
else:
    filename = video[0:len(video)-4]
    clip = moviepy.VideoFileClip(video)
    clip.write_videofile(filename + "_test.mp4")
