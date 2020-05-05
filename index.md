Our objective is to achieve higher compression in images by retaining only the essential features that are useful to a given computer vision system. For instance, given a computer vision system like VGG16 \cite{simonyan2014very} (a state of the art object detection neural network), the modified image compression technique should produce a compressed image that is sufficient enough for the VGG16 to detect the semantically important objects. We proposed two such approaches -

    * Depth map based JPEG encoder (d-JPEG)
    * Image Compression for VGG16 (VGG-Compression)

# Depth map based JPEG encoder (d-JPEG)


In the work by Prakash et al. \cite{prakash2017semantic}, they encode semantically-salient regions of the image with higher quality than the other regions. We try to extend this idea by using depth estimate map of the image. We use Dense depth \cite{alhashim2018high} to get the depth map to encode the regions that are closer to the camera with higher quality than the regions that are far away. The intuition for this approach is that the regions that are closer to the camera are likely to be more important for the end object detection system than the regions that are far away.



The motivation for this approach of using the depth based map instead of the existing approach using the semantically important region is that the depth estimation can be done with better accuracy and efficiency than finding out semantically important region in an image. There are lots of classical algorithms that are available to estimate the depth in the image which can be done with less compute power. However, to start with we have used a pre-trained model for depth estimation, and in the future we will try to compare it with the existing classical algorithms for depth estimation. 


The key contribution of this approach are,

* Use of depth map to do non uniform image compression on the image.
* Median filter on the depth map and using it along with the existing percentile based multi-level thresholding to create regions in the depth map.
* Doing the compression to the extent in which the important information near to the camera like text or numbers are still recognizable by the computer vision system.


## Design
<img width="331" alt="Screenshot 2020-05-04 at 16 21 04" src="https://user-images.githubusercontent.com/6566518/81014906-576d4580-8e23-11ea-8022-d61eecd1cbfb.png">


### Depth Estimator
We used the depth-estimator code \footnote{https://github.com/ialhashim/DenseDepth} and model from \cite{alhashim2018high} to generate a depth map for the given input image. It uses convolutional neural network (CNN) for computing a high-resolution depth map and the model that we used was trained using NYU Depth v2 dataset.

The input for the module is the original image and the output from this module is the depth map of the image in gray-scale.

### Median Filter

Before using the depth map for the image compression, we wrote an additional layer to smoothen the depth map using the median filter. This is used so that map has proper regions of varying depth, and it useful so that for these individual regions a uniform compression can be used.


For the following sections 2.3.3 and 2.3.4, we used the combiner code from \cite{prakash2017semantic} \footnote{https://github.com/iamaaditya/image-compression-cnn} with minor modifications.


### Image Combiner
The image combiner uses the list of compressed images with varied compression ratio to use that as  segment for various regions in the depth map.

We have tweaked the existing model based on percentile based multi-level threshold in the combiner for our needs to identify the segments in the depth and use the corresponding compressed image segment. 


## Evaluation for depth based compression (d-JPEG)

This table shows the file size savings got from the d-JPEG compression,
| MIN  | MAX | AVG |
| ----- | -- | -- |
| 3.05%  | 27.57% | 14.59% |

The average file size savings of **14.59%**.

This table shows the accuracy based on the EAST's text detection on d-JPEG compression,
| Metric | Value |
| ----- | -- |
| Accuracy  | 84.28% |




<table>
  <tr>
    <td>File Size reduction: 27.57% (Max)

Original File Size: 30 KB
Compressed File Size: 21.7 KB
</td>
    <td>File Size reduction: 3.05% (Min)
Original File Size:  19 KB
Compressed File Size: 18 KB
</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/6566518/81014677-e9c11980-8e22-11ea-88c4-5737ba195474.png" width=370 height=370></td>
    <td><img src="https://user-images.githubusercontent.com/6566518/81014681-ef1e6400-8e22-11ea-9774-8a3f8e39e7f7.png" width=370 height=370></td>
  
  </tr>
 </table>
 








