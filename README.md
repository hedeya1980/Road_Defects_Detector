# Pothole severity classification via computer vision
**Pothole Detection:**
*	We trained YOLOv7 model using a ‘Pothole Detection Dataset’.
*	The dataset consists of images from two different sources (it’s available at https://learnopencv.s3.us-west-2.amazonaws.com/pothole_dataset.zip):
**	The Roboflow pothole detection dataset.
**	Pothole dataset that is mentioned in this ResearchGate article – Dataset of images used for pothole detection.
*	After combining, the dataset now contains:
**	1265 training images
**	401 validation images
**	118 test images
*	We randomly chose 22 frames from the 2 Scenes of the competition dataset, annotated them, and included them in the training data.
*	We trained for 285 epochs, and achieved 0.671 mAP @0.5 IoU, and 0.38 @ 0.5:0.95 IoU
*	We run the detections on both of ‘Scene 1’, and ‘Scene 2’ folders, and the detection results can be found at https://drive.google.com/drive/folders/19I0-0FDI5cBLeHJOP8iWvtYYwRHOX2bI?usp=share_link and https://drive.google.com/drive/folders/1hoUwnIYCiV8LJQI_P8_XIJUbnms7B3GK?usp=share_link respectively.
*	The colab notebook can be found at https://colab.research.google.com/drive/1EkCaVemu3ms3FNdmtjuIp8_cLcxe0zXu?usp=share_link
*	The best weights are available at: https://drive.google.com/file/d/1-L5jLL70B8YmSRLPNM2gyFx6Sizt2G0I/view?usp=share_link

**3D Reconstruction:**
*	We used COLMAP (https://colmap.github.io/) for the 3D reconstruction.  
*	COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface. It offers a wide range of features for reconstruction of ordered and unordered image collections.
*	Due to the competition timing constraints, we applied it to Scene 2 frames from frame# 279 to frame# 412.
*	The results of Scene reconstruction were impressive as shown in the following figure for the resulting fused Point Cloud.
![Fused Point Cloud](https://drive.google.com/file/d/1ATqgaQadmv0iFXQASlIErxoM1_xMVaji/view?usp=share_link/scene.png)

**Potholes Monocular Depth Estimation:**
* We used the algorithm of "Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth" as a base to estimate the depth of the detected potholes
* The following google colab notebook shows how the algorithm could be used to estimate the depth
* https://colab.research.google.com/drive/1183Ak-zCc88ZlOVf--ou4XXgDIL5bC5Y?usp=share_link
* We took some potholes of the videos as a sample to check the algorithm     
