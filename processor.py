import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from collections import Counter
import sklearn.preprocessing as preprocessing
import png


images_path = "csv_images/"


def csv_looper():
    for root, dirs, files in os.walk("csv_images/"):
        for f in files:
            yield pandas.read_csv("{}/{}".format(images_path, f), header=None).as_matrix()


def save_as_image(image_matrix, name="temp.png"):
    oldmin, oldmax= image_matrix.min(), image_matrix.max()
    newmin, newmax= 0, 255
    image_matrix = (((image_matrix-oldmin)*(newmax-newmin))/(oldmax-oldmin)) + newmin
    image_matrix = map(lambda arr: map(lambda x: int(x), arr), image_matrix)
    png.fromarray(image_matrix, 'L').save(name)


csv_generator = csv_looper()
frame_1 = csv_generator.next()
frame_1[frame_1 < 30] = 0
save_as_image(frame_1)
frame_1 = frame_1[np.where(frame_1 != 0)].flatten()
features = np.array([frame_1.min(), frame_1.max(), np.mean(frame_1)])
features = np.append(features, features[1] - features[0])
bins_places = np.histogram(frame_1, np.linspace(30, 37.0, 101))[0]
features = np.concatenate((features, bins_places))
features = preprocessing.scale(features)
print features.mean(axis=0)

# bins = np.linspace(30, 37.0, 200)
# plt.hist(frame_1, bins)
# plt.savefig('hist.png', dpi=1000)
# plt.show()
