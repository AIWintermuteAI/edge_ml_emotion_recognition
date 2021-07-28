import argparse
import os
from tensorflow.keras.models import load_model, Model


def save_bottleneck(model_path, bottleneck_layer):
    bottleneck_weights_path = os.path.join(os.path.dirname(model_path),'bottleneck_weigths.h5')
    model = load_model(model_path)
    for layer in model.layers:
        if layer.name == bottleneck_layer:
            output = layer.output
    bottleneck_model = Model(model.input, output)
    bottleneck_model.save_weights(bottleneck_weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="path to trained model to extract bottleneck")
    parser.add_argument("--bottleneck_layer", type=str, default='global_average_pooling2d',
                        help="at which layer to 'cut' the model")
    args = parser.parse_args()
    save_bottleneck(args.model_path, args.bottleneck_layer)
