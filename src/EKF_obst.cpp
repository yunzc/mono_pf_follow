#include "EKF_obst.h" 

void EKFobst::load_initial(float f_x, float f_y, int cx, int cy, float deltat){
	fu = f_x; fv = f_y; // focal lengths
	u0 = cx; v0 = cy; // image center 
	dt = deltat; 
	state = 10*VectorXf::Ones(3); // initialize dummy
	covariance = 10*MatrixXf::Identity(3,3);

	// initialize Q and R 
	process_noise = 0.2;
	meas_noise = 0.2;
	Q = process_noise*MatrixXf::Identity(3,3);
	R = meas_noise*MatrixXf::Identity(2,2); // bounding box is decently accurate
}

void EKFobst::calculate_state_est(VectorXf& state_est){
	// state estimation according to motion model 
	state_est = state; // add random motion? 
}

void EKFobst::calculate_meas_est(VectorXf state_est, VectorXf& meas_est){
	// calculate measurement from state estimate 
	int num_obstacles = (state_est.size())/3;
	meas_est.setZero(num_obstacles*2);
	for (int i = 0; i < num_obstacles; i++){
		meas_est(2*i) = fu*state_est(3*i+1)/state_est(3*i) + u0; 
		meas_est(2*i+1) = fv*state_est(3*i+2)/state_est(3*i) + v0; 
	}
}

void EKFobst::calculate_F(MatrixXf& F){
	F = MatrixXf::Identity(state.size(), state.size());
}

void EKFobst::calculate_H(MatrixXf& H){
	int num_obstacles = (state.size())/3;
	H.setZero(num_obstacles*2, num_obstacles*3);
	// obstacle landmarks 
	for (int i = 0; i < num_obstacles; i++){
		H(2*i,3*i) = -fu*state(3*i+1)/(state(3*i)*state(3*i));
		H(2*i+1,3*i) = -fv*state(3*i+2)/(state(3*i)*state(3*i));
		H(2*i,3*i+1) = fu/state(3*i);
		H(2*i+1,3*i+2) = fv/state(3*i);
	}
}

void EKFobst::add_landmark(Vector3f pos, Matrix3f pos_covar){
	MatrixXf new_covar;
	new_covar.setZero(state.size()+3, state.size()+3); 
	new_covar.topLeftCorner(state.size(), state.size()) = covariance; 
	VectorXf new_state;
	new_state.setZero(state.size()+3); 
	new_state.head(state.size()) = state; 
	new_covar.bottomRightCorner(3,3) = pos_covar; 
	new_state.tail(3) = pos; 
	// assign to state
	state = new_state;
	covariance = new_covar;
	// also update Q and R 
	Q = dt*MatrixXf::Identity(state.size(), state.size());
	int num_obstacles = (state.size())/3;
	R = meas_noise*MatrixXf::Identity(2*num_obstacles, 2*num_obstacles);
}	

void EKFobst::delete_landmark(int indx){
	int curr_num_obstacles = state.size()/3;
	MatrixXf new_covar;
	new_covar.setZero(state.size()-3, state.size()-3); 
	new_covar.topLeftCorner(3*indx, 3*indx) = covariance.topLeftCorner(3*indx, 3*indx); 
	VectorXf new_state;
	new_state.setZero(state.size()-3); 
	new_state.head(3*indx) = state.head(3*indx); 
	int bc_size = new_state.size() - 3*indx;
	new_covar.bottomRightCorner(bc_size, bc_size) = covariance.bottomRightCorner(bc_size, bc_size);
	new_state.tail(bc_size) = state.tail(bc_size); 
	// assign to state 
	state = new_state; 
	covariance = new_covar;
	// also update Q and R
	Q = process_noise*MatrixXf::Identity(state.size(), state.size());
	int num_obstacles = state.size()/3;
	R = meas_noise*MatrixXf::Identity(2*num_obstacles, 2*num_obstacles);
}

void EKFobst::prediction(VectorXf& state_est, MatrixXf& P_est){
	calculate_state_est(state_est); // with motion update (prediction)
	// now calculate new estimated covariance P_est
	MatrixXf F; 
	calculate_F(F); // motion update matrix
	P_est = F*covariance*F.transpose() + Q;
}

void EKFobst::update(VectorXf measurement, VectorXf state_est, MatrixXf P_est){
	VectorXf h; 
	calculate_meas_est(state_est, h);
	VectorXf y = measurement - h; // difference between measurement and estimate
	MatrixXf H; 
	calculate_H(H);
	MatrixXf S = H*P_est*H.transpose() + R;
	MatrixXf K = P_est*H.transpose()*S.inverse(); // Kalman gain 
	state = state_est + K*y; // update state estimate in EKFobst 
	MatrixXf I = MatrixXf::Identity(P_est.rows(), P_est.cols());
	covariance = (I - K*H)*P_est;	// update covariance 
	std::cout << state << std::endl; 
}

void EKFobst::get_state(VectorXf& x){
	x = state;
}
void EKFobst::get_covariance(MatrixXf& P){
	P = covariance;
}