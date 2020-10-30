# Test script :)

To execute: `python warp_and_find_checkers.py <input_path> <output_path>`

Dependencies:
* Fire
* OpenCV
* Numpy
* Matplotlib

Questions: 
* 1- I expect that it runs acceptable and get similar results than test images.
* 2- 
  * If the board reference points are poorly labeled can crop part of some checkers, which will make dificult to find them. Can partially fix by padding reference points to outside of the board a bit.
  * Checkers with similar colors than pips can break it too. Apply edge detectors such as canny can help.
  * Shadows, blur, too dark/bright and etc. Use the current approach to label a bunch of images and create a dataset to a NN.
* 3- I'll move the image to HSV color space since is easier to diferentiate colors there. Having that, distinguish checkers would be a piece of cake :)
