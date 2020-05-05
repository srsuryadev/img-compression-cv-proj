# Image compression for computer vision system

Our objective is to achieve higher compression in images by retaining only the essential features that are useful to a given computer vision system. For instance, given a computer vision system like VGG16 \cite{simonyan2014very} (a state of the art object detection neural network), the modified image compression technique should produce a compressed image that is sufficient enough for the VGG16 to detect the semantically important objects. We proposed two such approaches -

    * Depth map based JPEG encoder (d-JPEG) - Compressed images retain information in the form perceivable by humans and computer vision system.
    * Image Compression for Object Detection - Information in the compressed images will be perceivable only by the target computer vision system. 


## Depth map based JPEG encoder (d-JPEG) 

We resused some of the modules from <a href="https://github.com/ialhashim/DenseDepth"> Dense Depth</a> and <a href = "https://github.com/iamaaditya/image-compression-cnn">Semantic Image compression</a> technique.
