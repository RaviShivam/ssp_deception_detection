import os
import subprocess
import pandas as pd

video_folder = "data_video/";
audio_folder = "data_audio/";
labels = pd.read_csv("deception_labels.csv", usecols=["id", "label"], dtype={"id": int, "label": str})
label_lookup = dict(map(lambda x: (x[0], x[1]), labels.as_matrix()))

def label_by_userstory(user, story):
    return int(label_lookup[user][story-1])

def remove_imls(data):
    walk = os.walk(data)
    root, dirs, files = walk.next()
    for _ in dirs:
        sroot,_,files = walk.next()
        for file in files:
            if (file[-3:]!="mp4"):
                os.remove(os.path.join(sroot, file))


walk = os.walk(video_folder)
root, dirs, files = walk.next()
for currdir in dirs:
    sroot,_,files = walk.next()
    subj = int(currdir.strip("subject_"))
    for file in files:
        story = int(file.strip(".mp4").split("_")[1])
        label = "truth" if label_by_userstory(subj, story)==1 else "lie"
        audio_path = os.path.join(audio_folder, label, file).strip(".mp4") + ".wav"
        extract_command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(os.path.join(sroot, file), audio_path)
        subprocess.call(extract_command, shell=True)