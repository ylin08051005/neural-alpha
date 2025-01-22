#!/bin/bash

if [ ! -f "data/tw_stock_data_2.zip" ]; then
    gdown https://drive.google.com/uc?id=17f0Yx4-i3tdnzxUBaR_16qvv5yOlNG85 -O data/tw_stock_data_2.zip
fi

if [ ! -d "data/tw_stock_data" ]; then
    mkdir data/tw_stock_data
    unzip data/tw_stock_data_2.zip -d data
    rm data/tw_stock_data_2.zip
fi