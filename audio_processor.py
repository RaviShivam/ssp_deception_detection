
from pyAudioAnalysis import audioTrainTest as aT
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
