#include "EKF_simple.h" 

void EKF::load_initial(float f_x, float f_y, int cx, int cy, float target_w, float target_h, float deltat){
	fu = f_x; fv = f_y; // focal lengths
	u0 = cx; v0 = cy; // image center 
	w = target_w; h = target_h; // shape of target 
	dt = deltat;
	state.setZero(6); // initialize as 0, no obsts (except for the x values)
	state(0) = 1; 
	covariance = 10*MatrixXf::Identity(6,6);
	// initialize Q and R 
	meas_noise = 0.1;
	Q = MatrixXf::Identity(6,6);
	Q.block(0,0,3,3) = dt*Matrix3f::Identity(3,3);
	R = meas_noise*MatrixXf::Identity(3,3); // bounding box is decently accurate TODO check if can set this
}

void EKF::calculate_state_est(VectorXf& state_est){
	// state estimation according to motion model 
	VectorXf newstate(state.size());
	newstate.segment(0,3) = state.segment(0,3) + state.segment(3,3)*dt;
	newstate.segment(3,3) = state.segment(3,3);
	state_est = newstate;
}
void EKF::calculate_meas_est(VectorXf state_est, VectorXf& meas_est){
	// calculate measurement from state estimate 
	meas_est.setZero(3);
	meas_est(0) = fu*(state_est(1))/state_est(0) + u0; 
	meas_est(1) = fv*(state_est(2))/state_est(0) + v0; 
	meas_est(2) = fv*h/state_est(0); // choose height so less vari
}

void EKF::calculate_F(MatrixXf& F){
	F.setZero(6,6);
	F.block(0,0,3,3) = Matrix3f::Identity(3,3);
	F.block(0,3,3,3) = Matrix3f::Identity(3,3);
	F.block(3,3,3,3) = Matrix3f::Identity(3,3);
}

void EKF::calculate_H(MatrixXf& H){
	H.setZero(3, 6);
	// first 6 cols all zeros
	// the target 
	H(0,0) = -fu*(state(1))/(state(0)*state(0));
	H(1,0) = -fv*(state(2))/(state(0)*state(0));
	H(2,0) = -fv*h/(state(0)*state(0));
	H(0,1) = fu/state(0);
	H(1,2) = fv/state(0);
}

void EKF::prediction(VectorXf& state_est, MatrixXf& P_est){
	calculate_state_est(state_est); // with motion update (prediction)
	// now calculate new estimated covariance P_est
	MatrixXf F; 
	calculate_F(F); // motion update matrix
	P_est = F*covariance*F.transpose() + Q; // TODO check matrix mult in eigen
	// std::cout << "predict: " << state(0) << " " << state_est(0) << std::endl; 
}

void EKF::update(VectorXf measurement, VectorXf state_est, MatrixXf P_est){
	VectorXf h_; 
	calculate_meas_est(state_est, h_);
	VectorXf y = measurement - h_; // difference between measurement and estimate
	std::cout << "state_est: " << state_est << std::endl; 
	// std::cout << "h: " << h << std::endl; 
	// std::cout << "meas: " << measurement << std::endl; 
	MatrixXf H; 
	calculate_H(H);
	MatrixXf S = H*P_est*H.transpose() + R; // TODO check matrix transpose/mult in eigen
	MatrixXf K = P_est*H.transpose()*S.inverse(); // Kalman gain 
	// std::cout << "K: " << K << std::endl; 
	state = state_est + K*y; // update state estimate in EKF 
	// std::cout << "Ky: " << K*y << std::endl; 
	MatrixXf I = MatrixXf::Identity(P_est.rows(), P_est.cols());
	covariance = (I - K*H)*P_est;	// update covariance 
	// std::cout << "update: " << state(0) << " " << state_est(0) << " " << h(0) << std::endl; 
}

void EKF::get_state(VectorXf& x){
	x = state;
}
void EKF::get_covariance(MatrixXf& P){
	P = covariance;
}