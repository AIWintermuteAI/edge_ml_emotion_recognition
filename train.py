import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import math
import logging
import argparse
import os

from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

from models.mobilenet_sipeed.mobilenet import MobileNet
from models.MobileFaceNet.MobileFaceNet import mobile_face_base
from utils import load_data
from data_generator import DataGenerator

logging.basicConfig(level=logging.DEBUG)

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for facial emotion estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_csv", type=str, required=True,
                        help="path to train database csv file")
    parser.add_argument("--validation_csv", type=str, required=True,
                        help="path to validation database csv file")                        
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--img_size", type=int, default=48,
                        help="image size")                        
    parser.add_argument("--nb_epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="adam",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="alpha parameter for filter number")
    parser.add_argument("--bottleneck_weights", type=str,
                        help="bottleneck_weights")
    parser.add_argument("--checkpoint", type=str,
                        help="checkpoint to continue training")
    parser.add_argument("--model_type", type=str, default="MobileFaceNet",
                        help="MobileNet or MobileFaceNet")

    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default=os.path.join("checkpoints", datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                        help="checkpoint dir")
    args = parser.parse_args()
    return args


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def get_model(img_size, alpha, num_classes = 5, model_type= 'MobileFaceNet', weights = None):

    logging.debug("Creating model...")
    input_image = Input(shape=(img_size, img_size, 3))

    if model_type == 'MobileNet':

        input_shapes_imagenet = [(128, 128,3), (160, 160,3), (192, 192,3), (224, 224,3)]
        input_shape = (128,128,3)
        for item in input_shapes_imagenet:
            if item[0] <= img_size[0]:
                input_shape = item

        mobilenet = MobileNet(input_shape=input_shape, input_tensor=input_image, 
                            alpha = alpha, weights = weights, 
                            include_top=False,
                            backend=tf.keras.backend, layers=tf.keras.layers, 
                            models=tf.keras.models, utils=tf.keras.utils)
        x = GlobalAveragePooling2D()(mobilenet.outputs[0])

    elif model_type == 'MobileFaceNet':

            x = mobile_face_base(input_shape=(img_size, img_size, 3), input_tensor=input_image, 
                            alpha = alpha, weights = weights, 
                            backend=tf.keras.backend, layers=tf.keras.layers, 
                            models=tf.keras.models, utils=tf.keras.utils)
            x = Flatten()(x)

    else:
        logging.error("Model type {} is not supported".format(model_type))

    x = Dropout(0.25)(x)

    predictions = Dense(num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs=input_image, outputs=predictions)

    initial_weights = [layer.get_weights() for layer in model.layers]
    if weights is not None:
        print('Transfer learning mode')
        model.load_weights(weights, by_name=True)
        for layer, initial in zip(model.layers, initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'Checkpoint contained no weights for layer {layer.name}!')

    return model

def main():
    args = get_args()
    train_dataset_path = args.train_csv
    validation_dataset_path = args.validation_csv
    checkpoint = args.checkpoint
    model_type = args.model_type
    batch_size = args.batch_size
    img_size = args.img_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    alpha = args.alpha
    use_augmentation = args.aug

    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.debug("Project folder: ".format(output_path))

    logging.debug("Loading data...")
    train_image_list, train_data = load_data(train_dataset_path)
    valid_image_list, valid_data = load_data(validation_dataset_path)

    num_classes = train_data.shape[1]

    train_num = len(train_image_list)
    test_num = len(valid_image_list)

    indexes = np.arange(train_num)
    np.random.shuffle(indexes)

    X_train = train_image_list[indexes]
    y_train = train_data[indexes]

    X_test = valid_image_list
    y_test = valid_data

    if checkpoint:
        model = load_model(checkpoint)
    else:
        model = get_model(img_size = img_size, alpha=alpha, num_classes = num_classes,
                                model_type = model_type, weights = args.bottleneck_weights)

    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='mean_absolute_error')

    logging.debug("Model summary...")
    model.summary()

    callbacks = [ReduceLROnPlateau(min_delta=1e-6, monitor="val_mean_absolute_error", verbose=2, mode="auto"),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_mean_absolute_error:.2f}.hdf5",
                                 monitor="val_mean_absolute_error",
                                 verbose=2,
                                 save_best_only=True,
                                 mode="auto"),
                EarlyStopping(monitor="val_mean_absolute_error", verbose=2, restore_best_weights=True, patience = 20)]

    logging.debug("Running training...")
        
    training_generator = DataGenerator(X_train, y_train, batch_size=batch_size, 
                                        img_size = img_size, prefix='data/processed_data/train/', augment = use_augmentation)()
    validation_generator = DataGenerator(X_test, y_test, batch_size=batch_size, 
                                        img_size = img_size, prefix='data/processed_data/valid/')()

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=train_num // batch_size,
                        validation_data=validation_generator,
                        validation_steps=test_num // batch_size,
                        epochs=nb_epochs, verbose=1,
                        callbacks=callbacks)

if __name__ == '__main__':
    main()
