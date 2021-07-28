#!/bin/bash
mkdir -p data
cd data

echo "Downloading processed FER+ dataset - the dataset ownership belongs to their respective authors. Consult README for license information and links."

if [ ! -f fer+repackaged.zip ]; then
    wget https://files.seeedstudio.com/ml/emotion/fer+repackaged.zip
fi

if [ ! -d processed_data ]; then
    unzip fer+repackaged.zip -d processed_data
fi

