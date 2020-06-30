# Lane Detection System

This repository consists of two projects:
1) Improving the quality of a low brightness video.
2) Lane detection and turn prediction
This projects aims to do a simple lane detection to mimic lane departure warning systems used in self-driving cars. 

### Improving the video quality:
- The various concepts that were tried are:
    - [Histogram Equalization](https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html)
    - [CLAHE (Contrast Limited Adaptive Histogram Equalization)](http://amroamroamro.github.io/mexopencv/opencv/clahe_demo_gui.html#3)
    - [Gamma Correction](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

### Lane Detection and turn prediction:
- This involved the concepts of **hough transform** and **hough lines**.


Implementation approach and explanation of the concepts involved and the solution can be found in the [report](./Report.pdf)

### Instructions to run the code:

##### For improving video quality problem:
[Dataset](https://drive.google.com/file/d/1IhaTYPnwTwEtj3VKnUq5f8C_Oeo6kgZ8/view)

- Go to directory: 'Code/improve_video_quality/'
- *$ python impove_image_quality.py*
- Please copy-paste the video 'Night Drive-2689.mp4' to the following location relative to this README file: './Code/media/improve_video_quality_data/'



##### For lane detection and turn prediction problem:
[Dataset](https://drive.google.com/drive/folders/1WL49qHfrsO7bB8rmsZkUNeF4c9t_abAz)

For dataset_1:
- Go to directory: 'Code/lane_detection/' 
- Run the file: *$ python lane_detection_images.py*
- Please copy-paste the images to the following location relative to this README file: './Code/media/lane_detection_data/images/'


For dataset_2:
- Go to directory: 'Code/lane_detection/'
- Run the file: *$ python lane_detection_video.py*
- Please copy-paste the images to the following location relative to this README file: './Code/media/lane_detection_data/video/'


### Output:

Improving video quality:
![alt text](./output/video_quality.PNG?raw=true "AR Tag")

Output videos can be found [here](https://drive.google.com/drive/folders/1CVt0Flg1HNlbpoh0rk_HrIXpVl4FV7Ct?usp=sharing)

