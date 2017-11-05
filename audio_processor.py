import os
import pickle
import pandas as pd
from pyAudioAnalysis import audioTrainTest as aT

audio_path = "data_audio"
all_files = []
labels = pd.read_csv("project_data/labels.csv", usecols=[0,1], dtype={"id": int, "label": str}, index_col=0)


def get_label_by_story(subject, story):
    return labels.loc[subject].values[0][story-1]

for root, dirs, files in os.walk(audio_path):
    for f in files:
        all_files.append(os.path.join(root, f))

predictions = {}
for trfile in all_files:
    originalname = trfile
    print originalname
    inter = originalname.split("/")[-1].strip("sub").strip(".wav").split("_")
    sub, st = int(inter[0]), int(inter[1])
    movedname = originalname.replace(originalname.split("/")[1], "fold")
    os.rename(originalname, movedname)
    aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "audio_models/svm/svm", True)
    prediction = aT.fileClassification(movedname, "audio_models/svm/svm", "svm")[1][1]
    prediction = 1 if prediction>0.5 else 0
    predictions[(sub, st)] = [prediction, get_label_by_story(sub, st)]
    os.rename(movedname, originalname)

with open("audio_prediction.pickle", "wb") as f:
    pickle.dump(predictions, f)
    # aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "audio_models/knn/knn", False)
    # aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "audio_models/randomforest/randomforest", False)
    # aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "audio_models/gradientboosting/gradientboosting", False)
    # aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "audio_models/extratrees/extratrees", False)

    # aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/knn/knn", "knn")
    # aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/randomforest/randomforest", "randomforest")
    # aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/gradientboosting/gradientboosting", "gradientboosting")
    # aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/extratrees/extratrees", "extratrees")
