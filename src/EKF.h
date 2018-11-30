#ifndef EKF_H
#define EKF_H

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

class EKF{
    VectorXf state; // consists of (in order): [vx vy vz wx wy wz xT yT zT dxT dyT dzT xi yi zi ... ]
    MatrixXf covariance; 
    MatrixXf Q; 
    MatrixXf R; 
    float dt; 
    float f; // camera focal length
    float w, h; //w, h target marker width and height (in physical world)
    int u0, v0; // image pixel center
    void calculate_state_est(VectorXf& state_est);
    void calculate_meas_est(VectorXf state_est, VectorXf& meas_est);
    void calculate_Rot(Matrix3f& R); // calculate rotation matrix 
    void calculate_F(MatrixXf& F);
    void calculate_H(MatrixXf& H);
    void calculate_dRdphi(Matrix3f& mat);
    void calculate_dRdthet(Matrix3f& mat);
    void calculate_dRdpsi(Matrix3f& mat);

public:
	void load_initial();
	void add_landmark(Vector3f pos, Matrix3f pos_covar);
    void delete_landmark(int indx); 
	void prediction(VectorXf& state_est, MatrixXf& P_est);
	void update(VectorXf measurement, VectorXf state_est, MatrixXf P_est);
	void get_state(VectorXf& state);
	void get_covariance(MatrixXf& P);
};

#endif