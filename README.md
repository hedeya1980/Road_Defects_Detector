# Pothole severity classification via computer vision
## Pothole Detection:
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

## 3D Reconstruction:
*	We used COLMAP (https://colmap.github.io/) for the 3D reconstruction.  
*	COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface. It offers a wide range of features for reconstruction of ordered and unordered image collections.
*	Due to the competition timing constraints, we applied it to Scene 2 frames from frame# 279 to frame# 412.
*	The results of Scene reconstruction were impressive as shown in the following figure for the resulting fused Point Cloud.
<p>
    <img src="https://raw.githubusercontent.com/hedeya1980/Images/main/scene.png" width="1000" height="520" />
</p>

## Proposed Pipeline:
1.	The 2D images are fed into the trained YOLOv7 model.
2.	The resulting bounding boxes are projected to the 3D point cloud to perform the following operations:
a.	Filtering the point cloud to the region corresponding to the 2D bounding box.
b.	Performing ‘Road Plane Segmentation’ for just the filtered region to decide which points belong to the road, and which ones belong to the pothole.
c.	Performing ‘Clustering’ for the off-road points, to determine the pothole clusters, and perform the required measurements/calculations.
3.	We perform the following measurements:
a.	We measure the Euclidean distance of pothole cluster points to the road plane (obtained in step 2-b above), to determine the maximum pothole depth.
b.	The cluster points are projected into the road plane to apply concave hull to the projected points and determine the pothole’s area.
c.	From the pothole’s area, we estimate the average diameter of the pothole.
d.	The pothole’s volume is estimated using the pothole’s area as well and distances of the pothole points from the road plane.
4.	We follow the following chart to classify Potholes into ‘Low’, ‘Medium’, and ‘High’ in terms of severity, and hence decide which segments of the road need urgent maintenance:
![Severity Classification Metrics](https://raw.githubusercontent.com/hedeya1980/Images/main/severity.png)
5.	The pipeline is illustrated in the following figure:
![Proposed Pipeline](https://raw.githubusercontent.com/hedeya1980/Images/main/pipeline.png)

## Prototype:
1.	In the submitted prototype, we illustrate the Point Cloud processing steps explained above.
2.	Due to timing instead of applying the 2D and 3D fusion as explained in the above proposed pipeline, we applied the plane segmentation to the whole point cloud. The full version will include the fusion of 2D bounding boxes and 3D point clouds.
3.	The segmented plane is illustrated in green color, the potholes are illustrated in yellow color, and the clusters that are higher than the plane are illustrated in light blue color.
4.	Although we used the metric units to record our pothole measurements, it’s worth mentioning that the units resulting from COLMAP are arbitrary units. So, the achieved measurements may need to be rescaled using objects from the scene with well known actual measurements (such as cars).
![Point Cloud After Processing](https://github.com/hedeya1980/Images/blob/main/processed_scene.png)
