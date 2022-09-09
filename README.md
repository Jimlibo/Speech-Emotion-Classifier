# Speech-Emotion-Classifier


![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)


## General
This repository contains an emotion classifying pipeline based on python scripts. The pipeline takes as input audio files, processes them and feeds them to a pretrained
model that will classify them based on what emotion they represent. The model was trained using approximately 600 audio files in Greek, each of them with a duration of 5 seconds. Due to the low number of samples, the accuracy of the model is about 75% which is not great, but not too bad either.

## Environment
The project was created using Anaconda. The virtual environment's configuration is stored inside conda_venv.yml file. In order to recreate that environment, use the command (inside anaconda prompt):
```sh
conda env create -f conda_venv.yml
```

## Data
The data used in this project were taken from [Acted Emotional Speech Dynamic Database] or AESDD for short. They can be found inside the [data] folder. To create the csv that will be used to train the model, execute the below command inside the Speech-Emotion-Classifier directory:
```sh
python3 preprocessing/audio_to_csv.py --src-dir data --dest-file converted/converted_audio.csv
```

## Training 
After creating the csv from the audio files, you can train the model by executing the command:
```sh
python3 training/model_training.py --src-file converted/converted_audio.csv --dest-file models/emotion_classifier.h5
```
In this repository, there is already a pretrained model inside the [models] folder, so the above step can be skipped. The optimal hyper parameters of the model, were choosen after an analysis that was perfomed. This analysis can be found inside Model_Tunning folder as a .ipynb file.

## Deployment
The deployment of the model is executed through main.py script. The script, takes as input an audio file, and a .h5 file. The latter one, contains a pretrained model
that will be used to predict the emotion of the audio file given. To run the script, use the following command:
```sh
python3 main.py --src-file my_audio_file.wav --model-file models/emotion_classifier.h5
```
The script will print the prediction, but if you want to save it in a file, you can use the '>' operator in UNIX-based systems

## License

MIT

**Free Software**


[Acted Emotional Speech Dynamic Database]: https://mega.nz/folder/0ShVXY7C#-73kVoK05OjTPEA95UUvMw
[data]: https://github.com/Jimlibo/Speech-Emotion-Classifier/tree/main/data
[models]: https://github.com/Jimlibo/Speech-Emotion-Classifier/tree/main/models
