import pandas as pd
import numpy as np
import pickle
import os
import sys
from hurry.filesize import size
from tqdm import tqdm
from termcolor import colored

thermal_folder = "thermal_images/"
timestamp_folder = "project_data/thermal_timestamps.csv"
label_folder = "project_data/labels.csv"

filter_temp = 32
total_num_frames = 40426

timestamps = pd.read_csv(timestamp_folder, index_col=0)
labels = pd.read_csv(label_folder, usecols=[0,1], dtype={"id": int, "label": str}, index_col=0)
game = pd.read_csv(label_folder, usecols=[0,1], dtype={"id": int, "game": int}, index_col=0)

def get_story_by_frame(subject, frame):
    second= frame/10.0 if (subject<15) else frame
    sub_data = timestamps.loc[subject].values
    if (second < sub_data[0]): return 0
    i, j = 0, 1
    story = -1
    for k in range(3):
        if (sub_data[i] <= second <= sub_data[j]):
            story = k+1
            break
        i += 2
        j += 2
    return story

def get_label_by_frame(subject, frame):
    story = get_story_by_frame(subject, frame)
    if (story == 0 or story == -1): return (story, -1)
    return (story, int(labels.loc[subject].values[0][story-1]))

def frame_with_info_traverse(thermal_path):
    walker = os.walk(thermal_path)
    walker.next() # skip the main dir
    for root, dirs, files in walker:
        if (len(files) == 0): continue # if the folder is empty or is root folder
        subject = int(root.split("/")[-2].strip("Subject")) # thermal_images/Subject10/CSV images => 10
        for f in files:
            frame_nr = int(f.split("_")[1].strip(".csv")) # FL-000020_369.csv => 369
            story, label = get_label_by_frame(subject, frame_nr) # get the story number and label of the frame
            if (story == -1): continue # skip if the frame is between the stories
            csv_frame = pd.read_csv(os.path.join(root, f)).as_matrix() # read the csv data
            yield (subject, story, frame_nr, csv_frame, label)

def extract_frame_features(frame_1):
    frame_1[frame_1 < filter_temp] = 0 #filter all pixels lower than 30
    frame_1 = frame_1[np.where(frame_1 != 0)].flatten() # new array with only filtered values
    if len(frame_1) == 0:
        return None
    features = np.array([frame_1.min(), frame_1.max(), np.mean(frame_1)]) # get mean, max and min values of pixels
    features = np.append(features, features[1] - features[0]) # add max difference in pixels
    bin_places = np.histogram(frame_1, np.linspace(filter_temp, 37.0, 101))[0] # take the values of the bins
    #     plot_bins(bin_places) # plot the bins if needed
    features = np.concatenate((features, bin_places)) # add primary features to bin features
    return features


csv_generator = frame_with_info_traverse(thermal_folder)
baseline_count = dict(zip(range(1, 25), np.zeros(24)))
baseline = dict(zip(range(1, 25), np.zeros(24)))
control_group = {}
test_group = {}
# for _ in tqdm(range(4760)):
#     csv_generator.next()

for _ in tqdm(range(total_num_frames)):
    subject, story, fnr, frame, label = csv_generator.next()
    # print subject, story, fnr, frame, label
    features = extract_frame_features(frame)
    if features is None:
        print colored((subject, story, fnr, frame, label), 'red')
        continue
    if (story==0):
        baseline_count[subject] = baseline_count[subject] + 1
        baseline[subject] = baseline[subject] + features
        continue
    update_dict = control_group if (game.loc[subject].values[0]==0) else test_group
    if ((subject, story) not in update_dict):
        update_dict[(subject, story)] = np.array([features, label])
    else:
        update_dict[(subject, story)] = np.vstack((update_dict[(subject, story)], [features, label]))


for key in baseline.keys():
    if baseline_count[key]==0: continue
    baseline[key] = baseline[key]/baseline_count[key]

def serialize(group, fi):
    with open(fi, 'wb') as f:
        pickle.dump(group, f)

serialize(control_group, "control_group.pickle")
serialize(test_group, "test_group.pickle")
serialize(baseline, "baseline.pickle")