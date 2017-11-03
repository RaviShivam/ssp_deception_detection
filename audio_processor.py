import os

from pyAudioAnalysis import audioTrainTest as aT
from glob import glob


def move_back(files):
    for filename in files:
        # putting the files back where they belong
        os.rename(filename[1], filename[0])

fold = os.getcwd() + os.path.sep + 'data_audio' + os.path.sep + 'fold'
lies = os.getcwd() + os.path.sep + 'data_audio' + os.path.sep + 'lie'
truth = os.getcwd() + os.path.sep + 'data_audio' + os.path.sep + 'truth'

# make folders if they are missing
directory = os.path.dirname(fold + os.path.sep)
print os.path.exists(directory)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + os.path.sep + 'lie')
    os.makedirs(directory + os.path.sep + 'truth')

try:
    for i in range(1, 25):
        # moving the files to mimic k-folds
        opperations = []
        for filepath in glob(lies + os.path.sep + 'sub' + str(i) + '_' + '*.wav'):
            a = (filepath, fold + os.path.sep + "lie" + os.path.sep + os.path.basename(filepath))
            os.rename(a[0], a[1])
            opperations.append(a)
        for filepath in glob(truth + os.path.sep + 'sub' + str(i) + '_' + '*.wav'):
            a = (filepath, fold + os.path.sep + 'truth' + os.path.sep + os.path.basename(filepath))
            os.rename(a[0], a[1])
            opperations.append(a)

        # training/modeling
        if glob(fold + os.path.sep + "lie" + os.path.sep + '*.wav') != []:
            aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "audio_models/svm/svm", False)
            aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "audio_models/knn/knn", False)
            aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "audio_models/randomforest/randomforest", False)
            aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "audio_models/gradientboosting/gradientboosting", False)
            aT.featureAndTrain(["data_audio/lie","data_audio/truth"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "audio_models/extratrees/extratrees", False)

            aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/svm/svm", "svm")
            aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/knn/knn", "knn")
            aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/randomforest/randomforest", "randomforest")
            aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/gradientboosting/gradientboosting", "gradientboosting")
            aT.fileClassification("data_audio/truth/sub1_1.wav", "audio_models/extratrees/extratrees", "extratrees")

        # move back the unused files
        move_back(opperations)
except:
    move_back(opperations)
# delete models?
