import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from collections import Counter
import sklearn.preprocessing as preprocessing
import png

images_path = "csv_images/"
timestamps = pd.read_csv("project_data/thermal_timestamps.csv", index_col=0)
labels = pd.read_csv("project_data/labels.csv", usecols=[0,1], dtype={"id": int, "label": str}, index_col=0)
# labels
def get_story_by_frame(subject, frame):
    second= frame/10.0 if (subject<15) else frame
    sub_data = timestamps.loc[subject].values
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
    if (story == -1): return None
    return labels.loc[subject].values[0][story-1]



def csv_looper():
    for root, dirs, files in os.walk("csv_images/"):
        for f in files:
            yield pd.read_csv("{}/{}".format(images_path, f), header=None).as_matrix()


def save_as_image(image_matrix, name="temp.png"):
    oldmin, oldmax= image_matrix.min(), image_matrix.max()
    newmin, newmax= 0, 255
    image_matrix = (((image_matrix-oldmin)*(newmax-newmin))/(oldmax-oldmin)) + newmin
    image_matrix = map(lambda arr: map(lambda x: int(x), arr), image_matrix)
    png.fromarray(image_matrix, 'L').save(name)
# csv_generator = csv_looper()
# frame_1 = csv_generator.next()

frame_1 = pd.read_csv("SSP-000009_20.csv", header=None).as_matrix()
frame_1[frame_1 < 30] = 0
save_as_image(frame_1, name="arthur_filter.png")
# frame_1 = frame_1[np.where(frame_1 != 0)].flatten()
# features = np.array([frame_1.min(), frame_1.max(), np.mean(frame_1)])
# features = np.append(features, features[1] - features[0])
# bins_places = np.histogram(frame_1, np.linspace(30, 37.0, 101))[0]
# features = np.concatenate((features, bins_places))
# features = preprocessing.scale(features)
# print features.mean(axis=0)

# bins = np.linspace(30, 37.0, 200)
# plt.hist(frame_1, bins)
# plt.savefig('hist.png', dpi=1000)
# plt.show()



