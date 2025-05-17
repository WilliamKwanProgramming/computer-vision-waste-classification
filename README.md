# computer-vision-waste-classification

Building a real time waste classification vision model that tells you the type of waste (Recyclable, Hazardous, Food, and Residual)

## Model

Uses a ResNet50 convolutional backbone—pretrained on the ImageNet dataset. The ResNet50’s original top layer is fed into a Global Average Pooling layer to produce a compact feature vector. That vector is then passed through a 100-unit Dense layer with ReLU activation to learn task-specific representations, followed by a 9-unit Dense layer with softmax activation that outputs probability scores over the nine target classes. 

## Requirements
Python 3+, Tensorflow, OpenCV, CVZone
