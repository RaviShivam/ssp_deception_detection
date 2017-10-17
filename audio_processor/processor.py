import subprocess
import os


for root, dirs, files in os.walk("../video_files/"):
    for f in files:
        command = "ffmpeg -i {}{} -ab 160k -ac 2 -ar 44100 -vn ../audio_files/{}.wav".format(root, f, f)
        subprocess.call(command, shell=True)

