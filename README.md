# Image compression for computer vision system

Our objective is to achieve higher compression in images by retaining only the essential features that are useful to a given computer vision system. For instance, given a computer vision system like Inception v3 (a state of the art object detection neural network), the modified image compression technique should produce a compressed image that is sufficient enough for the Inception v3 to detect the semantically important objects. We propose two such approaches -

    * Depth map based JPEG encoder (d-JPEG) - Compressed images retain information in the form perceivable by humans and computer vision system.
    * Image Compression for Object Detection - Information in the compressed images will be perceivable only by the target computer vision system. 


## Depth map based JPEG encoder (d-JPEG) 

We resused some of the modules from <a href="https://github.com/ialhashim/DenseDepth"> Dense Depth</a> and <a href = "https://github.com/iamaaditya/image-compression-cnn">Semantic Image compression</a> technique.


### Requirements

Model - <a href ="https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5">NYU Depth V2 </a> download and place it in the models directory.

### Run
To generate the depth maps for the images in the examples dir,

```
python3 test.py 
```

The generated depth-map files will be stored in the depth-map dir.

To compress the input images using depth map,

```
python3 combine_images.py
```

You can find the image files compressed using depth map and jpeg (original) in the output dir.

## Object detection based Image Compression

We used parts of code from [pytorch-image-comp-rnn](https://github.com/1zb/pytorch-image-comp-rnn) for Full Resolution Image Compression network and [PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) for Inception v3 trained model. Dataset was obtained from [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html). 

### Requirements

```
imageio==2.8.0
numpy==1.18.3
torch==1.5.0
torchvision==0.5.0+cu92
```

### Run
```
Training
python train.py -f <path/to/dataset>

Evaluation
python encoder.py --model <path/to/encoder/checkpoint> --input <path/to/dataset> --output <path/to/save/output/compressed> --true <path/to/save/true/images>

```
