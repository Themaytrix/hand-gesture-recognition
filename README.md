# hand-gesture-recognition
a hand gesture recognition using mediapipe hand pose estimation

## Files
- `test.py` a sample program for inferencing
- `keyclassifier.ipynb` jupyter notebook for preprocessing,building neural network, training model and inference testing
- `label.csv` csv file containing labels for the training data
- `keypoint.csv` contains the logged landmarks
- `keypoint_classifier.py` module for inferencing

## Walkthrough

### Keying keypoints
`test.py` sample program allows you to key in landmarks from the MediaPipe hand estimation into a `keypoint.csv` file. 
![mediapipe hand estimation](https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png)

- #### how?
  - When you run the program, there's a prompt that tells you to press "n" which sets `mode` to 1 which is the keying mode
  - You get to choose between (0-9) as the classifying labels for the data points collected. i.e say you make a peace sign and in the `label.csv` your peace sign is first on the list, you press (0) when keying the points(due to indexing)
  - the keypoints are collected into `keypoint.csv` file


### Training

