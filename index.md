Our objective is to achieve higher compression in images by retaining only the essential features that are useful to a given computer vision system. For instance, given a computer vision system like VGG16 \cite{simonyan2014very} (a state of the art object detection neural network), the modified image compression technique should produce a compressed image that is sufficient enough for the VGG16 to detect the semantically important objects. We proposed two such approaches -

    * Depth map based JPEG encoder (d-JPEG)
    * Image Compression for VGG16 (VGG-Compression)

# Depth map based JPEG encoder (d-JPEG)**


In the work by Prakash et al. \cite{prakash2017semantic}, they encode semantically-salient regions of the image with higher quality than the other regions. We try to extend this idea by using depth estimate map of the image. We use Dense depth \cite{alhashim2018high} to get the depth map to encode the regions that are closer to the camera with higher quality than the regions that are far away. The intuition for this approach is that the regions that are closer to the camera are likely to be more important for the end object detection system than the regions that are far away.



The motivation for this approach of using the depth based map instead of the existing approach using the semantically important region is that the depth estimation can be done with better accuracy and efficiency than finding out semantically important region in an image. There are lots of classical algorithms that are available to estimate the depth in the image which can be done with less compute power. However, to start with we have used a pre-trained model for depth estimation, and in the future we will try to compare it with the existing classical algorithms for depth estimation. 


The key contribution of this approach from us are,

* Use of depth map to do non uniform image compression on the image.
* Median filter on the depth map and using it along with the existing percentile based multi-level thresholding to create regions in the depth map.
* Doing the compression to the extent in which the important information near to the camera like text or numbers are still recognizable by the computer vision system.








