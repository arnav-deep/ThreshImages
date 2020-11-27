# ThreshImages

Python scripts for thresholding images using common thresholding methods.

Instead of directly using the library, here, the functions for threshold coding are directly written for each method in each script.

## Setup

Open termina in project directory and install the requirements.

```python
pip install -r requirements.txt
```

The requirements are only used for basic functionality's like reading the image, converting it to black and white, displaying the image, and a simple histogram function for Otsu's method.

## The Threshhold Coding Methods

The following threshold methods have been applied here.

### Li's iterative Minimum Cross Entropy Threshold Method

This method has been applied in [```thresh_lis.py```](https://github.com/arnav-deep/ThreshImages/blob/main/thresh_lis.py).

  * How to run:
  
      ```python
      python thresh_lis.py <image_filename>
      ```
  
   * Note: All other arguments except the username imagefile are optional and have pre-defined values in their respective script.

### Local Threshold Method

This method has been applied in [```thresh_local.py```](https://github.com/arnav-deep/ThreshImages/blob/main/thresh_local.py).

All 4 methods have been applied (default is using gaussian) —

    * Generic
    * Gaussian
    * Mean
    * Median
 
  * How to run:
  
      ```python
      python thresh_local.py <image_filename> <method>
      ```

### Mean Threshold Method

This method has been applied in [```thresh_mean.py```](https://github.com/arnav-deep/ThreshImages/blob/main/thresh_mean.py).

  * How to run:
  
      ```python
      python thresh_mean.py <image_filename>
      ```

### Minimum Threshold Method

This method has been applied in [```thresh_minimum.py```](https://github.com/arnav-deep/ThreshImages/blob/main/thresh_minimum.py).

  * How to run:
  
      ```python
      python thresh_minimum.py <image_filename>
      ```

### Otsu's Threshold Method

This method has been applied in [```thresh_otsu.py```](https://github.com/arnav-deep/ThreshImages/blob/main/thresh_otsu.py).

  * How to run:
  
      ```python
      python thresh_otsu.py <image_filename> <nbins>
      ```
      * Note: Here, ```nbins``` is the number of bins used to calculate histogram.

### Yen's Threshold Method

This method has been applied in [```thresh_yen.py```](https://github.com/arnav-deep/ThreshImages/blob/main/thresh_yen.py).

  * How to run:
  
      ```python
      python thresh_yen.py <image_filename> <ndim>
      ```
      * Note: Here, ```nbins``` is the number of bins used to calculate histogram.
