import os

video_folder = "data_video/";
audio_folder = "data_audio/";

walk = os.walk(video_folder)
root, dirs, files = walk.next()

for _ in dirs:
    sroot,_,files = walk.next()
    for file in files:
        if (file[-3:]!="mp4"):
            os.remove(os.path.join(sroot, file))

