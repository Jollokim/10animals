# from https://github.com/Jollokim/Alexnet-edgemaps-vs-RGB
import argparse
import os
import shutil

import numpy as np

from sklearn.model_selection import train_test_split

"""
    A script for splitting natural images (NI) dataset into train, valid and test sets.
    This wasn't done initial
    train: 70% valid: 20% test: 10%
    randomly picking images for each set
"""


def get_args_parser():
    parser = argparse.ArgumentParser('NI train valid test splitter', add_help=False)

    # directory parameters:
    parser.add_argument('--root_dir', type=str, required=True, help='root dir of NI dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir')

    return parser


def main(args):
    # defining different path where images can be found and where the split should end up

    root_dir = args.root_dir
    output_dir = args.output_dir

    sets = ['train', 'valid', 'test']

    # getting classes from dataset
    labels = os.listdir(root_dir)

    # list for holding image paths and correspond class
    img_list = []
    label_list = []

    # fills above list by going through each class folder
    for label in labels:
        for img in os.listdir(f'{root_dir}/{label}'):
            img_list.append(f'{img}')
            label_list.append(label)

    # Turning the above list into numpy arrays, to speed things a little
    X = np.array(img_list)
    y = np.array(label_list)

    print(X.shape)

    # splits dataset into 70% train and 30% valid/test
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.30, random_state=1)

    # then splits the 30% valid/test to 20% valid and 10% test. Done by making 66% of the 30% valid/test the valid and
    # the rest 33% of the 30% valid/test makes the test 10%
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.33, random_state=1)

    if not os.path.exists(f'{output_dir}'):
        os.mkdir(f'{output_dir}')

    for s in sets:
        # creates the train, valid, test split inside the new dataset folder
        if not os.path.exists(f'{output_dir}/{s}'):
            os.mkdir(f'{output_dir}/{s}')

            # creates class folders inside the set folder
            for label in labels:
                os.mkdir(f'{output_dir}/{s}/{label}')

    # copies images to their corresponding class folder of the train split
    for i in range(len(X_train)):
        shutil.copyfile(f'{root_dir}/{y_train[i]}/{X_train[i]}', f'{output_dir}/train/{y_train[i]}/{X_train[i]}')

    # copies images to their corresponding class folder of the valid split
    for i in range(len(X_valid)):
        shutil.copyfile(f'{root_dir}/{y_valid[i]}/{X_valid[i]}', f'{output_dir}/valid/{y_valid[i]}/{X_valid[i]}')

    # copies images to their corresponding class folder of the test split
    for i in range(len(X_test)):
        shutil.copyfile(f'{root_dir}/{y_test[i]}/{X_test[i]}', f'{output_dir}/test/{y_test[i]}/{X_test[i]}')


if __name__ == '__main__':
    # creates commandline parser
    parser = argparse.ArgumentParser('Create NI train valid test split', parents=[get_args_parser()])
    args = parser.parse_args()

    # passes the commandline arguments to the create_variation function
    main(args)