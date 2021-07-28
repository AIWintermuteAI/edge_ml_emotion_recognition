import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_args():
    parser = argparse.ArgumentParser(description="This script evaluate face emotion estimation model "
                                                 "using the FER+ test data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_file = args.model_file

    # load model and weights
    img_size = 48
    batch_size = 64
    model = load_model(model_file)

    dataset_root = Path(__file__).parent.joinpath("../data")

    test_image_dir = dataset_root.joinpath("processed_data/test")
    test_csv_path = dataset_root.joinpath("test.csv")

    df = pd.read_csv(str(test_csv_path))
    df = df.drop(['Usage', 'NF'], axis=1)
    
    predictions = []

    faces = np.empty((batch_size, img_size, img_size, 3))

    for i, row in tqdm(df.iterrows()):
        image_path = dataset_root.joinpath(test_image_dir, row['Image name'])

        img = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /=  255.
        img -=  0.5
        img *=  2.
        faces[i % batch_size] = img

        if (i + 1) % batch_size == 0 or i == len(df) - 1:
            results = model.predict(faces)
            predictions.extend(results)

    class_abs_error = 0.0
    accuracy = 0

    for i, row in df.iterrows():
        class_abs_error += abs(predictions[i] - df.iloc[i, 2:]/10)
        top_result = np.argmax(predictions[i])
        label = np.argmax(df.iloc[i, 2:])
        accuracy += 1 if top_result == label else 0     

    print("Top result accuracy: {}".format(accuracy / len(df)))
    class_abs_error = class_abs_error / len(df)
    total_abs_error = sum(class_abs_error)/len(class_abs_error)
    print("Class-wise MAE: \n{}".format(class_abs_error))
    print("Total MAE: {}".format(total_abs_error))

if __name__ == '__main__':
    main()
