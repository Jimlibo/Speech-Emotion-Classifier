"""
Created 9 September 2022
@author: JL
Description: Script that uses a pretrained model to assign an audio file to one of 5 emotion categories

Usage:
    1) Take as input an audio file and use a pretrained model stored in model-file, to recognize what emotion is present
    in the audio file and print it
        python3 main.py
        --src-file <.wav file>
        --model-file <.h5 file>

Example:
    python3 main.py
        --src-file my_audio.wav
        --model-file emotion_classifier.h5
"""

import os
import argparse
import numpy as np
from keras.models import load_model
from preprocessing.audio_to_csv import extract_features


class EmotionClassifier:
    def __init__(self, src_file=None, model_file=None):
        # source file checks
        if src_file is None:
            print("src_file is required!")
            exit(1)
        elif not os.path.exists(src_file):
            print("src_file {} does not exist!".format(src_file))
            exit(1)
        else:
            self.src_file = src_file

        # model file checks
        if model_file is None:
            print("model_file is required!")
            exit(1)
        elif not os.path.exists(model_file):
            print("model_file {} does not exist!".format(model_file))
            exit(1)
        else:
            self.model = load_model(model_file)

        self.input = extract_features(self.src_file, mfcc=True, chroma=True, mel=True)

    def get_prediction(self):
        # preparing input
        input_vector = np.concatenate(list(map(np.array, self.input)))
        input_vector = np.array([input_vector])

        label_decoding = ['anger', 'disgust', 'fear', 'happiness', 'sadness']   # for int-to-string decoding

        # getting the prediction
        model_output = self.model.predict(input_vector, verbose=0)[0]   # get the output of the model
        pred_as_int = np.argmax(model_output)  # get the neuron (index) that has the highest score
        final_prediction = label_decoding[pred_as_int]   # decode the prediction to get a string value

        print("The audio file given represents:", final_prediction)


def parse_input(args=None):
    """
    param args: The command line arguments provided by the user
    :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-file", type=str, action="store", metavar="src_file",
                        required=True, help="The file that will be used to train the model")
    parser.add_argument("-m", "--model-file", type=str, action="store", metavar="model_file",
                        required=True, help="The file with the pretrained model that will be used")

    return parser.parse_args(args)


def main(args):
    classifier = EmotionClassifier(src_file=args.src_file, model_file=args.model_file)
    classifier.get_prediction()


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
