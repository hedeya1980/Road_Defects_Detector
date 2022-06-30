// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"
#include <unordered_set>

using namespace std;


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::ProjectCloud(typename pcl::PointCloud<PointT>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficients)
{
    typename pcl::PointCloud<PointT>::Ptr projected_cloud(new pcl::PointCloud<PointT>());
    // Project the model inliers
    pcl::ProjectInliers<PointT> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    // proj.setIndices (inliers);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coefficients);
    proj.filter(*projected_cloud);

    return projected_cloud;
}

template<typename PointT>
void ProcessPointClouds<PointT>::ConcaveHullCloud(typename pcl::PointCloud<PointT>::Ptr cloud, typename pcl::PointCloud<PointT>::Ptr cloud_hull)
{
    pcl::ConcaveHull<PointT> chull;
    chull.setInputCloud(cloud);
    chull.setAlpha(0.1);
    chull.reconstruct(*cloud_hull);
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering
    /*
    typename pcl::PointCloud<PointT>::Ptr filteredCloud (new pcl::PointCloud<PointT> ());
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(filterRes,filterRes,filterRes);
    sor.filter(*filteredCloud);
    std::cerr << "Voxel Grid filtering: Point cloud was downsampled to " << filteredCloud->points.size () << " data points" << std::endl;
    */
    typename pcl::PointCloud<PointT>::Ptr regionCloud (new pcl::PointCloud<PointT> ());
    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    //region.setInputCloud(filteredCloud);
    region.setInputCloud(cloud);
    region.filter(*regionCloud);
    std::cerr << "ROI filtering: Point cloud was downsampled to " << regionCloud->points.size () << " data points" << std::endl;
    /*
    typename std::vector<int> indices;
    pcl::CropBox<PointT> roof(true);
    roof.setMin(Eigen::Vector4f(-1.5,-1.7,-1,1));
    roof.setMax(Eigen::Vector4f(2.6,1.7,-0.4,1));
    roof.setInputCloud(regionCloud);
    roof.filter(indices);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    for(int point:indices)
        inliers->indices.push_back(point);
    
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(regionCloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*regionCloud);

    */
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return regionCloud;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
  // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr obstCloud (new pcl::PointCloud<PointT> ());
    typename pcl::PointCloud<PointT>::Ptr planeCloud (new pcl::PointCloud<PointT> ());

    for (int index : inliers -> indices)
        planeCloud -> points.push_back(cloud -> points[index]);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);
    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::RansacPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol)
{
    auto startTime = std::chrono::steady_clock::now();
	std::unordered_set<int> inliersResult;
    //pcl::PointIndices::Ptr inliersResult (new pcl::PointIndices());
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 
    while (maxIterations--)
    {
        // Randomly sample subset and fit line
        std::unordered_set<int> inliers;
        //pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
        while (inliers.size() < 3)
            inliers.insert(rand() % (cloud->points.size()));
        
        float x1, y1, z1, x2, y2, z2, x3, y3, z3;

        auto itr = inliers.begin();
        x1 = cloud->points[*itr].x;
        y1 = cloud->points[*itr].y;
        z1 = cloud->points[*itr].z;
        itr++;
        x2 = cloud->points[*itr].x;
        y2 = cloud->points[*itr].y;
        z2 = cloud->points[*itr].z;
        itr++;
        x3 = cloud->points[*itr].x;
        y3 = cloud->points[*itr].y;
        z3 = cloud->points[*itr].z;
        //float a = y1 - y2;
        float i = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
        float a = i;
        //float b = x2 - x1;
        float j = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1);
        float b = j;
        //float c = x1 * y2 - x2 * y1;
        float k = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        float c = k;
        float d = -1*(i * x1 + j * y1 + k * z1);
        // Measure distance between every point and fitted line
        for (int index = 0; index<cloud->points.size(); index++)
        {
            if (inliers.count(index) > 0)
                continue;

            PointT point = cloud->points[index];
            //float x3 = point.x;
            float x4 = point.x;
            //float y3 = point.y;
            float y4 = point.y;
            float z4 = point.z;

            float dist = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);

            // If distance is smaller than threshold count it as inlier

            if (dist <= distanceTol)
                inliers.insert(index);
        }
        
        // Return indicies of inliers from fitted line with most inliers
        if (inliers.size() > inliersResult.size())
            inliersResult = inliers;
    }

    pcl::PointIndices::Ptr pclInliers(new pcl::PointIndices());
    for (auto i:inliersResult)
    {
        pclInliers->indices.push_back(i);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Ransac took " << elapsedTime.count() << " milliseconds" << std::endl;
	
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ransacResult = SeparateClouds(pclInliers,cloud);
    return ransacResult;

}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold, pcl::ModelCoefficients::Ptr coefficients)
{
    /*
    auto startTime = std::chrono::steady_clock::now();
	std::unordered_set<int> inliersResult;
    //pcl::PointIndices::Ptr inliersResult (new pcl::PointIndices());
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 
    while (maxIterations--)
    {
        // Randomly sample subset and fit line
        std::unordered_set<int> inliers;
        //pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
        while (inliers.size() < 3)
            inliers.insert(rand() % (cloud->points.size()));
        
        float x1, y1, z1, x2, y2, z2, x3, y3, z3;

        auto itr = inliers.begin();
        x1 = cloud->points[*itr].x;
        y1 = cloud->points[*itr].y;
        z1 = cloud->points[*itr].z;
        itr++;
        x2 = cloud->points[*itr].x;
        y2 = cloud->points[*itr].y;
        z2 = cloud->points[*itr].z;
        itr++;
        x3 = cloud->points[*itr].x;
        y3 = cloud->points[*itr].y;
        z3 = cloud->points[*itr].z;
        //float a = y1 - y2;
        float i = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
        float a = i;
        //float b = x2 - x1;
        float j = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1);
        float b = j;
        //float c = x1 * y2 - x2 * y1;
        float k = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
        float c = k;
        float d = -1*(i * x1 + j * y1 + k * z1);
        // Measure distance between every point and fitted line
        for (int index = 0; index<cloud->points.size(); index++)
        {
            if (inliers.count(index) > 0)
                continue;

            PointT point = cloud->points[index];
            //float x3 = point.x;
            float x4 = point.x;
            //float y3 = point.y;
            float y4 = point.y;
            float z4 = point.z;

            float dist = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);

            // If distance is smaller than threshold count it as inlier

            if (dist <= distanceThreshold)
                inliers.insert(index);
        }
        
        // Return indicies of inliers from fitted line with most inliers
        if (inliers.size() > inliersResult.size())
            inliersResult = inliers;
    }

    pcl::PointIndices::Ptr pclInliers(new pcl::PointIndices());
    for (auto i:inliersResult)
    {
        pclInliers->indices.push_back(i);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Ransac took " << elapsedTime.count() << " milliseconds" << std::endl;
	
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ransacResult = SeparateClouds(pclInliers,cloud);
    return ransacResult;
    */

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
    // TODO:: Fill in this function to find inliers for the cloud.

    //pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    
    //pcl::PointIndices::Ptr inliners (new pcl::PointIndices());
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setInputCloud(cloud);
    seg.segment(*inliers,*coefficients);
    if(inliers->indices.size()==0)
    {
        std::cout<<"Could not estimate a planar model for the given dataset!"<<std::endl;
    }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentCircle2D(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold, pcl::ModelCoefficients::Ptr coefficients)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    // TODO:: Fill in this function to find inliers for the cloud.

    //pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    //pcl::PointIndices::Ptr inliners (new pcl::PointIndices());
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CIRCLE2D);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0)
    {
        std::cout << "Could not estimate a planar model for the given dataset!" << std::endl;
    }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Circle segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);
    return segResult;
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (clusterTolerance); // 2cm
    ec.setMinClusterSize (minSize);
    ec.setMaxClusterSize (maxSize);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    for (pcl::PointIndices getIndices : cluster_indices)
    {
        typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);
        for (int index : getIndices.indices)
            cloudCluster->points.push_back(cloud->points[index]);
        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;
        clusters.push_back(cloudCluster);
    }


    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

template<typename PointT>
BoxQ ProcessPointClouds<PointT>::BoundingBoxQ(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    BoxQ box;
    // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    //pcl::compute3DCentroid(*cloud_projected, pcaCentroid);
    pcl::compute3DCentroid(*cluster, pcaCentroid);
    Eigen::Matrix3f covariance;
    //computeCovarianceMatrixNormalized(*cloud_projected, pcaCentroid, covariance);
    computeCovarianceMatrixNormalized(*cluster, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));  /// This line is necessary for proper orientation in some cases. The numbers come out the same without it, but
                                                                                    ///    the signs are different and the box doesn't get correctly oriented in some cases.
    /* // Note that getting the eigenvectors can also be obtained via the PCL PCA interface with something like:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloudSegmented);
    pca.project(*cloudSegmented, *cloudPCAprojection);
    std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;
    std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;
    // In this case, pca.getEigenVectors() gives similar eigenVectors to eigenVectorsPCA.
    */
    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
    typename pcl::PointCloud<PointT>::Ptr cloudPointsProjected(new pcl::PointCloud<PointT>);
    //pcl::transformPointCloud(*cloud_projected, *cloudPointsProjected, projectionTransform);
    pcl::transformPointCloud(*cluster, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    // Final transform
    const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA); //Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
    const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
    
    box.bboxQuaternion = bboxQuaternion;
    box.bboxTransform = bboxTransform;
    box.x_min = minPoint.x, box.y_min = minPoint.y, box.z_min = minPoint.z;
    box.x_max = maxPoint.x, box.y_max = maxPoint.y, box.z_max = maxPoint.z;
    box.cube_length = std::abs(maxPoint.x - minPoint.x);
    box.cube_width = std::abs(maxPoint.y - minPoint.y);
    box.cube_height = std::abs(maxPoint.z - minPoint.z);
    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}

template<typename PointT>
void ProcessPointClouds<PointT>::saveBin(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::save(file, *cloud);
    std::cerr << "Saved " << cloud->points.size() << " data points to " + file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}



template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}