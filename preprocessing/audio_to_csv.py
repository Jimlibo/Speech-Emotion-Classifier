"""
Created 30 August 2022
@author: JL
Description: Script that transforms all audio files into a csv file

Usage:
    Take all *.wav files from a directory containing subdirectories, extract  specific features from each audio file
    and store them to a csv file
        python3 audio_to_csv.py
        --src-dir <folder>
        --dest-file <.csv file>

Example:
    python3 audio_to_csv.py
        --src-dir data
        --dest-file converted_audio.csv
"""

import argparse
import os
import soundfile
import librosa
import librosa.display
import numpy as np
import pandas as pd
from datetime import datetime


def extract_features(file_name, **kwargs):  # helping function
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    with soundfile.SoundFile(file_name) as sound_file:
        x = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = []

        if chroma or contrast:
            stft = np.abs(librosa.stft(x))

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
            result.append(mfccs)

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result.append(chroma)

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T, axis=0)
            result.append(mel)

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result.append(contrast)

        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate).T, axis=0)
            result.append(tonnetz)

    return result


class AudioConverter:
    def __init__(self, src_dir=None, dest_file=None):
        # checks about source directory
        if src_dir is None:
            print("src_dir is required!")
            exit(1)
        elif not os.path.exists(src_dir) and src_dir is not None:
            print("src_dir {} does not exist!".format(src_dir))
            exit(1)
        else:
            self.src_dir = src_dir

        # checks about destination file
        if dest_file is None:
            print("dest_file is required!")
            exit(1)
        else:
            self.dest_file = dest_file

        self.df = None   # dataframe to store the converted audio
        self.subdirs = None  # will be used to store subdirectories of src_dir

    def get_subdirs(self):  # fill a list with all the subdirectories contain in src_dir
        print("Gathering subdirectories")
        self.subdirs = os.listdir(self.src_dir)

        return self

    def construct_df(self):
        print("Constructing dataframe")

        mfcc, chroma, mel = [], [], []   # lists for each feature extracted from the audio file
        labels = []  # list with the label for each audio file
        for sd in self.subdirs:
            cur_dir = os.path.join(self.src_dir, sd)  # current directory

            for file in os.listdir(cur_dir):  # for each audio file in the specific subdirectory
                try:   # get the desired features
                    [mf, c, me] = extract_features(os.path.join(cur_dir, file), mfcc=True, chroma=True, mel=True)

                    # updating the appropriate lists
                    mfcc.append(list(map(str, mf)))
                    chroma.append(list(map(str, c)))
                    mel.append(list(map(str, me)))
                    labels.append(sd)
                except:
                    continue

        # turn arrays into strings
        mfcc = list(map(lambda x: '~'.join(x), mfcc))
        chroma = list(map(lambda x: '~'.join(x), chroma))
        mel = list(map(lambda x: '~'.join(x), mel))

        # create the dataframe
        self.df = pd.DataFrame({
            'mfcc': mfcc, 'chroma': chroma, 'mel': mel, 'label': labels
        })

        return self

    def export(self):
        print("Exporting to {}".format(self.dest_file))
        self.df.to_csv(self.dest_file, index=False)

        return self

    def pipeline(self):
        self.get_subdirs().construct_df().export()


def parse_input(args=None):
    """
    param args: The command line arguments provided by the user
    :return:  The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src-dir", type=str, action="store", metavar="src_dir",
                        required=True, help="The directory with subdirectories with audio files")
    parser.add_argument("-d", "--dest-file", type=str, action="store", metavar="dest_file",
                        required=True, help="The destination file where the generated csv will be saved")

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    audio_converter = AudioConverter(src_dir=args.src_dir, dest_file=args.dest_file)
    audio_converter.pipeline()

    print("Script execution time: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    ARG = parse_input()
    main(ARG)
