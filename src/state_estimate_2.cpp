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
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>

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
    void add_new_feature(cv::Point2f pt);

public:
	StateEstimator(float f_x, float f_y, int cx, int cy, float w, float h, float dt);
};

StateEstimator::StateEstimator(float f_x, float f_y, int cx, int cy, float w, float h, float freq){
	// load values 
	filt.load_initial(f_x, f_y, cx, cy, w, h, 1.0/freq);
    filt_obst.load_initial(f_x, f_y, cx, cy, 1.0/freq);
	fu = f_x; fv = f_y;
	u0 = cx; v0 = cy; 
	yz_uncertainty = 0.05;
	init_dist = 2.0;
	max_init_dist = 5.0; 
	maxFeatures = 88;

    // define publisher 
	target_pub = nh.advertise<geometry_msgs::PointStamped>("target_pos", 2);
	obstacle_pub = nh.advertise<sensor_msgs::PointCloud2> ("/cloud", 100);
	
    // start video streaming 
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

        // initialize a vector of zeros to keep track of matches 
        std::vector<int> matched(featurePts.size(), 0); 

        // filtering 
        MatrixXf P_est, P_obst_est; VectorXf state_est, state_obst_est; 
        // motion prediction step 
        filt.prediction(state_est, P_est); 
        filt_obst.prediction(state_obst_est, P_obst_est);
        // build the measurement vector 
        VectorXf meas = VectorXf::Zero(3);
        int num_obsts = (state_obst_est.size()-3)/3; 
        VectorXf meas_obs = VectorXf::Zero(num_obsts*2); 
        MatrixXf R_obs = MatrixXf::Zero(num_obsts*2,num_obsts*2); // noise matrix
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
        }else{
            filt.motion_update(state_est, P_est);
            // // add to feature points 
            // float u = fu*state_est(1)/state_est(0) + (float)u0; 
            // float v = fv*state_est(2)/state_est(0) + (float)v0; 
            // float len = fv*h/state_est(0); 
            // featurePts.push_back(cv::Point2f(u,v));
            // featurePts.push_back(cv::Point2f(u-len/3,v+len/3));
            // featurePts.push_back(cv::Point2f(u-len/3,v-len/3));
            // featurePts.push_back(cv::Point2f(u+len/3,v+len/3));
            // featurePts.push_back(cv::Point2f(u+len/3,v-len/3));
        }

        // some visuals 
        for( size_t i = 0; i < featurePts.size(); i++ ){
            int radius = 5;
            circle(src, featurePts[i], radius, cv::Scalar(0, 0, 255));
        }

        std::vector<int> to_delete; // keep track of features to delete 
        // now need to fill in the obstacle meas info 
        for (int i = 0; i < num_obsts; i++){
            int u = fu*state_obst_est(3+3*i+1)/state_obst_est(3+3*i) + u0; 
            int v = fv*state_obst_est(3+3*i+2)/state_obst_est(3+3*i) + v0;
            // extract corresponding covariance matrix 
            Matrix3f P_obst = P_obst_est.block(3+3*i,3+3*i,3,3);
            // get eigenvalue/vectors
            EigenSolver<Matrix3f> es(P_obst);
            MatrixXcf D = es.eigenvalues().asDiagonal(); // egienvector matrix
            MatrixXcf V = es.eigenvectors(); // col vect with the eigenvalue
            float x_val = (float(state_obst_est(3*i)));
            // get the variances from eigenvalue matrix
            float x_var = (float)D(0,0).real();
            float y_var = (float)D(1,1).real(); 
            float z_var = (float)D(2,2).real();
            float r = sqrt(MAX(y_var, z_var)); // two times largest std dev
            // float x_min = MAX(0.1, x_val - sqrt(x_var));
            // calculate min dist acceptable
            float min_dst_sq = fu*fv*r*r/(x_val*x_val);
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
                R_obs.block(2*i,2*i,2,2) = min_dst_sq*10*Matrix2f::Identity(2,2);
                // mark as matched 
                matched[indx] = 1; 
            }else{
                // delete if uncertainty too high 
                if (x_var > init_dist || y_var > yz_uncertainty || z_var > yz_uncertainty){
                    // mark for deleteion
                    to_delete.push_back(i);
                }
                meas_obs(2*i) = u; 
                meas_obs(2*i+1) = v; 
                R_obs.block(2*i,2*i,2,2) = 10e6*Matrix2f::Identity(2,2);
            }
        }
        // std::cout << "num obs: " << num_obsts << std::endl; 
        filt_obst.update(meas_obs, state_obst_est, P_obst_est, R_obs);

        // delete features 
        std::reverse(to_delete.begin(),to_delete.end());
        for (int i = 0; i < to_delete.size(); i++){
            filt_obst.delete_landmark(to_delete[i]);
        }
        // then add new features (if features after deletion less than certain threshold)
        if (num_obsts < maxFeatures){
            for (int i = 0; i < featurePts.size(); i++){
                if (matched[i] == 0){
                    add_new_feature(featurePts[i]);
                }
            }
        }

        // ros publishing
        // add target to Point message 
        VectorXf target_state; 
        filt.get_state(target_state);
        geometry_msgs::PointStamped target_pt;
        target_pt.header.frame_id = "/base_link";
        target_pt.header.stamp = ros::Time::now(); 
        target_pt.point.x = target_state(0); target_pt.point.y = target_state(1);
        target_pt.point.z = target_state(2);
        target_pub.publish(target_pt); // publish target 

        // add points to pointcloud 
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        VectorXf obstalces_state; 
        filt_obst.get_state(obstalces_state); 
        std::cout << "dxdydzy: " << obstalces_state(0) << " " << obstalces_state(1) << " " << obstalces_state(2) << std::endl; 
        int n = (obstalces_state.size()-3)/3; 
        for (int i = 0; i < n; i++){
            pcl::PointXYZ pt; 
            pt.x = obstalces_state(3+3*i); 
            pt.y = obstalces_state(3+3*i+1);
            pt.z = obstalces_state(3+3*i+2);
            cloud->push_back(pt);
        }
        // create header 
        sensor_msgs::PointCloud2 pc2; 
        pc2.header.frame_id = "/base_link";
        pc2.header.stamp = ros::Time::now();
        cloud->header = pcl_conversions::toPCL(pc2.header);
        obstacle_pub.publish(cloud);
        // viz
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

void StateEstimator::add_new_feature(cv::Point2f pt){
    // from new feature (2d image pt) add a set of possible 3d points into prob map
    int u = pt.x - u0; 
    int v = pt.y - v0; 
    // create unit vector correspondinng to direction of feature pt
    float hx = 1; float hy = u/fu; float hz = v/fv;
    float mag = hx/sqrt(hx*hx + hy*hy + hz*hz);
    hx = hx/mag; hy = hy/mag; hz = hz/mag; 
    float dist = init_dist; 
    while (dist < max_init_dist){
        Vector3f p(dist*hx, dist*hy, dist*hz);
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
        filt_obst.add_landmark(p, PC); // args are (pos, pos_covar)
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