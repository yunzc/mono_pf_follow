#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cmath>
#include <math.h> 
#include <Eigen/Dense>

struct landmark{
    cv::Point2f pt; 
    int hits; // this will be turn into probability later on
};

int max_hits = 10; 
cv::Mat src, src_gray;
int maxFeatures = 88; // number of tracked feature every cycle 
cv::RNG rng(12345);
const char* source_window = "Image";

// currently tracked points
std::vector<landmark> tracked_pts; 

// shi-tomasi feature detection
void detect_features(std::vector<cv::Point2f>& pts){
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

void find_closest_landmark(landmark obj, std::vector<cv::Point2f> detected, int& indx){
    // find the landmark closest to current from the detected 
    // return -1 if none is close enough 
    double cx = obj.pt.x; double cy = obj.pt.y;
    indx = -1; 
    double min_dst = 10*(double)max_hits/(double)obj.hits;
    for (int i = 0; i < detected.size(); i++){
        // calculate dist 
        double lx = detected[i].x; double ly = detected[i].y;
        double dist = sqrt((cx - lx)*(cx - lx) + (cy - ly)*(cy - ly));
        if (dist < min_dst){
            indx = i;
            min_dst = dist; 
        }
    }
}

void update(std::vector<cv::Point2f> detected_pts, std::vector<landmark>& tracked_pts){
    for (int i = 0; i < tracked_pts.size(); i++){
        int closest; 
        find_closest_landmark(tracked_pts[i], detected_pts, closest);
        if (closest != -1){
            // update point
            tracked_pts[i].pt = detected_pts[closest];
            tracked_pts[i].hits = MAX(max_hits, tracked_pts[i].hits + 1);
            detected_pts.erase(detected_pts.begin() + closest); // remove point from detected pts list
        }
    }
    for (int j = 0; j < detected_pts.size(); j++){
        // add the remainder to tracked_pts
        landmark newl; 
        newl.pt = detected_pts[j];
        newl.hits = 1;
        tracked_pts.push_back(newl);
    }
}

int main(int argc, char** argv){
    // initiate vid capture
    cv::VideoCapture inputVideo;
    inputVideo.open(0);
    // initiate aruco dictionary 
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(src);
        // color conversion for shi tomasi (grayscale)
        cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
        // aruco detection
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        // mask the region where the aruco is  
        if (ids.size() > 0){
            cv::aruco::drawDetectedMarkers(src, corners, ids);
            for (int i = 0; i < ids.size(); i++){
                std::vector<cv::Point> polypts; 
                for (int j = 0; j < corners[i].size(); j++){
                    polypts.push_back(cv::Point((int)corners[i][j].x, (int)corners[i][j].y));
                }
                fillConvexPoly(src_gray, polypts, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
                // mask target from feature detector
            }
        }
        std::vector<cv::Point2f> featurePts;
    	detect_features(featurePts); // shi tomasi feature detector 
        update(featurePts, tracked_pts);
        int factor = 5;
        // note that even with mask, there will be features detected on the corners of the
        // target marker. Hopefully that will be fine (only 4 pts)
        for( size_t i = 0; i < tracked_pts.size(); i++ ){
            int radius = max_hits/tracked_pts[i].hits;
            int hits = tracked_pts[i].hits;
            circle(src, tracked_pts[i].pt, factor*radius, cv::Scalar(250 - 25*hits, 0, 5 + 25*hits));
        }
        cv::namedWindow(source_window);
        cv::imshow( source_window, src );
    	char key = (char) cv::waitKey(1);
        if (key == 27)
            break;
    }
    return 0;
}