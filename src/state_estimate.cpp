#include <cmath>
#include <math.h> 
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h> 

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/aruco.hpp>

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <pcl_ros/point_cloud.h>

#include "EKF.h"
#include <Eigen/Dense> // for matrices and vectors
#include <Eigen/Eigenvalues>

using namespace Eigen;

typedef pcl::PointCloud<pcl::PointXYZ> Cloud; 
typedef pcl::PointXYZ Point; 

class StateEstimator{
	ros::NodeHandle nh;
	ros::Publisher target_pub; 
	ros::Publisher obstacle_pub;
	EKF filt; 
	cv::Mat src, src_gray;
	int ID = 0; // target labeled with aruco code of ID = 0
	int maxFeatures;
	float init_dist; // when a new feature detected, initiate along semi-infinite line: choose spacing along this line
	float max_init_dist; // (which will also impact the initialized covariance)
	float yz_uncertainty; // uncertainty (stand dev) in yz plane when adding new feature
	float fu, fv; // focal length (init this)
	int u0, v0; // image center 
	const char* source_window = "Image";
	void detect_features(std::vector<cv::Point2f>& pts);
	void add_new_features(std::vector<cv::Point2f>& pts);

public:
	StateEstimator(float f_x, float f_y, int cx, int cy, float w, float h, float dt);
};

StateEstimator::StateEstimator(float f_x, float f_y, int cx, int cy, float w, float h, float freq){
	// load values 
	filt.load_initial(f_x, f_y, cx, cy, w, h, 1.0/freq);
	fu = f_x; fv = f_y;
	u0 = cx; v0 = cy; 
	yz_uncertainty = 0.5;
	init_dist = 1.0;
	max_init_dist = 5.0; 
	maxFeatures = 88;
	target_pub = nh.advertise<geometry_msgs::Point>("target_pos", 2);
	obstacle_pub = nh.advertise<Cloud>("obstacle_cloud", 10);
	cv::VideoCapture inputVideo;
	inputVideo.open(0);
	// initiate aruco dictionary 
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    ros::Rate rate(freq);
    while (inputVideo.grab() && nh.ok()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(src);
        // color conversion for shi tomasi (grayscale)
        cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
        // aruco detection
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        std::vector<cv::Point2f> target_corners;// corners of our target 
        bool target_detected = false; 
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        // mask the region where the aruco is  
        if (ids.size() > 0){ // if target is detected 
            cv::aruco::drawDetectedMarkers(src, corners, ids);
            for (int i = 0; i < ids.size(); i++){
            	if (ids[i] == ID){
            		target_corners = corners[i]; 
            		target_detected = true;
            		break;
            	}
            }
        }
        if (target_detected){
            std::vector<cv::Point> polypts; 
            // std::cout << "target corners: " << std::endl; 
            for (int j = 0; j < target_corners.size(); j++){
            	// std::cout << target_corners[j] << std::endl; 
                polypts.push_back(cv::Point((int)target_corners[j].x, (int)target_corners[j].y));
            }
            fillConvexPoly(src_gray, polypts, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
            // mask target from feature detector
        }

        // shi tomasi feature detection
        std::vector<cv::Point2f> featurePts;
    	detect_features(featurePts); // shi tomasi feature detector 
        for( size_t i = 0; i < featurePts.size(); i++ ){
            int radius = 5;
            circle(src, featurePts[i], radius, cv::Scalar(0, 0, 255));
        }
        MatrixXf P_est; VectorXf state_est; 
        filt.prediction(state_est, P_est); // motion prediction step 
        int num_obsts = (state_est.size() - 12)/3;
        std::cout << "number of obst points: " << num_obsts << std::endl; 
        // build the measurement vector 
        VectorXf meas = VectorXf::Zero(4+2*num_obsts);
        // first fill in the bounding box: 
        if (target_detected){
        	// upper left corner (negative negative)
        	meas(0) = (float)target_corners[2].x; 
        	meas(1) = (float)target_corners[2].y;
        	// lower right corner (post post)
        	meas(2) = (float)target_corners[0].x; 
        	meas(3) = (float)target_corners[0].y;
        }
        // now need to fill in the obstacle meas info 
        for (int i = 0; i < num_obsts; i++){
        	int u = fu*state_est(12+3*i+1)/state_est(12+3*i) + u0; 
			int v = fv*state_est(12+3*i+2)/state_est(12+3*i) + v0; 
			// extract corresponding covariance matrix 
			Matrix3f P_obst = P_est.block(12+3*i,12+3*i,3,3);
			// get eigenvalue/vectors
			EigenSolver<Matrix3f> es(P_obst);
			MatrixXcf D = es.eigenvalues().asDiagonal(); // egienvector matrix
			MatrixXcf V = es.eigenvectors(); // col vect with the eigenvalues
        }
        cv::namedWindow(source_window);
        cv::imshow( source_window, src );
        ros::spinOnce(); 
        rate.sleep();
    	char key = (char) cv::waitKey(1);
        if (key == 27)
            break;
    }
}

void StateEstimator::detect_features(std::vector<cv::Point2f>& pts){
	maxFeatures = MAX(maxFeatures, 1);   
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    cv::goodFeaturesToTrack( src_gray,
                         pts,
                         maxFeatures,
                         qualityLevel,
                         minDistance,
                         cv::Mat(),
                         blockSize,
                         gradientSize,
                         useHarrisDetector,
                         k );
}

void StateEstimator::add_new_features(std::vector<cv::Point2f>& pts){
	// from new feature (2d image pt) add a set of possible 3d points into prob map 
	for (int i = 0; i < pts.size(); i++){
		int u = pts[i].x - u0; 
		int v = pts[i].y - v0; 
		// create unit vector correspondinng to direction of feature pt
		float hx = 1; float hy = u/fu; float hz = v/fv;
		float mag = hx/sqrt(hx*hx + hy*hy + hz*hz);
		hx = hx/mag; hy = hy/mag; hz = hz/mag; 
		float dist = 0.0; 
		while (dist < max_init_dist){
			Vector3f pt(dist*hx, dist*hy, dist*hz);
			Matrix3f PC, V, D; // calculate covariance from eigenvectors and eigenvalues 
			dist += init_dist;
			D.setZero(3,3); 
			D(0,0) = (init_dist/2)*(init_dist/2);
			// set up eigenvalues
			D(1,1) = yz_uncertainty*yz_uncertainty; D(2,2) = yz_uncertainty*yz_uncertainty;
			// set up eigenvector matrix
			V(1,1) = 1; // [0; 1; 0] 
			V(2,2) = 1; // [0; 0; 1]
			V(0,0) = hx; V(1,0) = hy; V(2,0) = hz; // [hx; hy; hz]
			PC = V*D*V.transpose(); 
			filt.add_landmark(pt, PC); // args are (pos, pos_covar)
		}
	}
}
// pcl::PointXYZ toPush;
//             toPush.x = px; toPush.y = py; toPush.z = pZ;

//             combined.points.push_back(toPush);

int main(int argc, char** argv){
	ros::init(argc, argv, "state_estimator");
	StateEstimator vision(atof(argv[1]), atof(argv[2]), 
			atoi(argv[3]), atoi(argv[4]), atof(argv[5]), atof(argv[6]), atof(argv[7]));
}

// TODO: I think just coding the update/motion model 