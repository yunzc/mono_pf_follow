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

#include "EKF_simple.h"
#include "EKF_obst.h"
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
    EKFobst filt_obst; 
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
    filt_obst.load_initial(f_x, f_y, cx, cy, 1.0/freq);
	fu = f_x; fv = f_y;
	u0 = cx; v0 = cy; 
	yz_uncertainty = 0.5;
	init_dist = 2.0;
	max_init_dist = 5.0; 
	maxFeatures = 44;
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
        // place mask on target 
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

        MatrixXf P_est, P_obst_est; VectorXf state_est, state_obst_est; 
        // motion prediction step 
        filt.prediction(state_est, P_est); 
        filt_obst.prediction(state_obst_est, P_obst_est);
        // build the measurement vector 
        VectorXf meas = VectorXf::Zero(3);
        int num_obsts = state_obst_est.size()/3; 
        VectorXf meas_obs = VectorXf::Zero(num_obsts*2); 

        // first fill in the bounding box: 
        if (target_detected){
            float x = 0; 
            float y = 0; 
            std::vector<float> heights; 
            for (int j = 0; j < target_corners.size(); j++){
                x += (float)target_corners[j].x;
                y += (float)target_corners[j].y;
                // std::cout << target_corners[j] << std::endl;
                if (j == 1){
                    float dx = (float)target_corners[j].x - (float)target_corners[j+1].x; 
                    float dy = (float)target_corners[j].y - (float)target_corners[j+1].y;
                    heights.push_back(sqrt(dx*dx + dy*dy));
                }else if (j == 3){
                    float dx = (float)target_corners[j].x - (float)target_corners[0].x; 
                    float dy = (float)target_corners[j].y - (float)target_corners[0].y;
                    heights.push_back(sqrt(dx*dx + dy*dy));
                }
            }
        	meas(0) = x/(float)target_corners.size();
        	meas(1) = y/(float)target_corners.size();
            meas(2) = (heights[0] + heights[1])/2.0; 
            // update
            filt.update(meas, state_est, P_est);
        }

        // now need to fill in the obstacle meas info 
        for (int i = 0; i < num_obsts; i++){
            int u = fu*state_obst_est(3*i+1)/state_obst_est(3*i) + u0; 
            int v = fv*state_obst_est(3*i+2)/state_obst_est(3*i) + v0; 
            // extract corresponding covariance matrix 
            Matrix3f P_obst = P_obst_est.block(3*i,3*i,3,3);
            // get eigenvalue/vectors
            EigenSolver<Matrix3f> es(P_obst);
            MatrixXcf D = es.eigenvalues().asDiagonal(); // egienvector matrix
            MatrixXcf V = es.eigenvectors(); // col vect with the eigenvalues 
            float min_dst_sq = 100;
            int indx = -1;
            for (int j = 0; j < featurePts.size(); j++){
                float dist_sq = (featurePts[j].x - (float)u)*(featurePts[j].x - (float)u)
                    + (featurePts[j].y - (float)v)*(featurePts[j].y - (float)v);
                if (dist_sq < min_dst_sq){
                    min_dst_sq = dist_sq; 
                    indx = j;
                }
            }
            if (indx != -1){
                meas_obs(2*i) = featurePts[indx].x; 
                meas_obs(2*i+1) = featurePts[indx].y; 
                featurePts.erase(featurePts.begin()+indx);
            }
        }
        filt_obst.update(meas_obs, state_obst_est, P_obst_est);

        // then add new features 
        if (num_obsts < 1){
            add_new_features(featurePts);
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
        float dist = init_dist; 
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
            filt_obst.add_landmark(pt, PC); // args are (pos, pos_covar)
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