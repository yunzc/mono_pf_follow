#ifndef EKFOBST_H
#define EKFOBST_H

// define feature patch class 
// encodes not only its image but also it estimated location 

#include <iostream>
#include <cmath>
#include <math.h> 
#include <fstream>
#include <string>
#include <stdlib.h> 
#include <Eigen/Dense>

using namespace Eigen;

class EKFobst{
    VectorXf state; // consists of (in order): [vx vy vz wx wy wz xT yT zT dxT dyT dzT xi yi zi ... ]
    MatrixXf covariance; 
    MatrixXf Q; 
    MatrixXf R; 
    float dt; 
    float fu, fv; // camera focal length 
    float w, h; //w, h target marker width and height (in physical world)
    int u0, v0; // image pixel center
    float process_noise;
    float meas_noise; 
    void calculate_state_est(VectorXf& state_est);
    void calculate_meas_est(VectorXf state_est, VectorXf& meas_est);
    void calculate_F(MatrixXf& F);
    void calculate_H(MatrixXf& H);

public:
	void load_initial(float fu, float fv, int u0, int v0, float dt);
	void add_landmark(Vector3f pos, Matrix3f pos_covar);
    void delete_landmark(int indx); 
	void prediction(VectorXf& state_est, MatrixXf& P_est);
	void update(VectorXf measurement, VectorXf state_est, MatrixXf P_est, MatrixXf R);
	void get_state(VectorXf& state);
	void get_covariance(MatrixXf& P);
};

#endif