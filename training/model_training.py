"""
Created 3 September 2022
@author: JL
Description: Script that uses a .csv file  as input to train an ANN model and save it to a .h5 file

Usage:
    Take a csv file  that was generated from audio_to_csv.py and use it to create and
    train a Neural Network, that will be saved to a specified file :
        python3 model_training.py
        --src-file <.csv file>
        --dest-file <.h5 file>

Example:
    python3 model_training.py
        --src-file converted_audio.csv
        --dest-file audio_nn.h5
"""

import os
import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from datetime import datetime
from sklearn.model_selection import train_test_split


def convert_to_float_array(lst):   # helping function
    float_l = list(map(float, lst))
    return np.array(float_l)


def create_model(d1, d2, dr, input_shape):
    model = keras.models.Sequential()
    model.add(layers.Dense(d1, input_shape=(input_shape,)))
    model.add(layers.Dense(d2, activation='relu'))
    model.add(layers.Dropout(dr))
    model.add(layers.Dense(5))  # the output layer, it contains 5 neurons , since it has 5 possible categories

    return model


class ModelTrainer:
    def __init__(self, src_file=None, dest_file=None):
        if src_file is None:
            print("src_file is required!")
            exit(1)
        elif not os.path.exists(src_file):
            print("src_file {} does not exist!".format(src_file))
            exit(1)
        else:
            self.src_file = src_file

        # checks about destination file
        if dest_file is None:
            print("dest_file is required!")
            exit(1)
        else:
            self.dest_file = dest_file

        self.df = pd.read_csv(self.src_file)
        self.features = None   # list for features that will be used as input to the model
        self.input_shape = None  # input shape for the model
        self.labels = None   # list with the labels of the previous features
        self.model = None   # model that will be trained

    def prepare_features(self):
        print('Preparing features')

        self.df['features'] = self.df['mfcc'] + '~' + self.df['chroma'] + '~' + self.df['mel']  # concat all columns

        features_as_string = list(self.df['features'])  # get the new column, containing all the features
        features_as_float = list(map(lambda x: x.split('~'), features_as_string))  # convert from string to list
        self.features = list(map(convert_to_float_array, features_as_float))  # convert elements of list to float
        self.input_shape = self.features[0].shape[0]

        return self

    def prepare_labels(self):
        print('Preparing labels')

        labels = list(self.df['label'])

        # encode the labels
        label_encoding = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'sadness': 4
        }
        self.labels = list(map(lambda x: label_encoding[x], labels))

        return self

    def costruct_model(self):
        print('Creating the model')

        self.model = create_model(16, 64, 0.25, self.input_shape)  # creating the model with the optimal parameters
        self.model.compile(  # compiling the model
            optimizer='adam',  # optimizer function
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # loss/cost function
            metrics=['accuracy']  # metrics that we are interested in
        )

        return self

    def train_model(self):
        print('Training the model')

        # splitting data to training and testing
        self.features, self.labels = np.array(self.features), np.array(self.labels)
        train_data, test_data, train_labels, test_labels = train_test_split(self.features, self.labels, train_size=0.7)

        # training the model
        self.model.fit(train_data, train_labels, epochs=100, validation_data=(test_data, test_labels), verbose=0)

        # evaluate the model and print the accuracy that was achieved
        test_loss, test_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)
        print('Accuracy of the model:', test_accuracy)

        return self

    def save(self):
        print('Saving model to {}'.format(self.dest_file))

        self.model.save(self.dest_file)
        return self

    def pipeline(self):
        self.prepare_features().prepare_labels().costruct_model().train_model().save()


def parse_input(args=None):
    """
    param args: The command line arguments provided by the user
    :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-file", type=str, action="store", metavar="src_file",
                        required=True, help="The file that will be used to train the model")
    parser.add_argument("-d", "--dest-file", type=str, action="store", metavar="dest_file",
                        required=True, help="The destination file where the trained model will be saved")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    trainer = ModelTrainer(src_file=args.src_file, dest_file=args.dest_file)
    trainer.pipeline()

    print("Script execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
