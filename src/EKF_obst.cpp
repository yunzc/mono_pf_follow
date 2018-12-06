#include "EKF_obst.h" 

void EKFobst::load_initial(float f_x, float f_y, int cx, int cy, float deltat){
	fu = f_x; fv = f_y; // focal lengths
	u0 = cx; v0 = cy; // image center 
	dt = deltat; 
	state = VectorXf::Zero(3); // initialize dummy
	covariance = MatrixXf::Identity(3,3);

	// initialize Q 
	process_noise = 0.01;
	Q = process_noise*MatrixXf::Identity(3,3);
}

void EKFobst::calculate_state_est(VectorXf& state_est){
	// state estimation according to motion model 
	state_est = state; // add random motion?
	int num_obsts = (state.size() - 3)/3; 
	for (int i = 0; i < num_obsts; i++){
		state_est.segment(3+3*i,3) = state.segment(3+3*i,3) + dt*state.segment(0,3);
	}
}

void EKFobst::calculate_meas_est(VectorXf state_est, VectorXf& meas_est){
	// calculate measurement from state estimate 
	int num_obstacles = (state_est.size() - 3)/3;
	meas_est.setZero(num_obstacles*2);
	for (int i = 0; i < num_obstacles; i++){
		meas_est(2*i) = fu*state_est(3+3*i+1)/state_est(3+3*i) + u0; 
		meas_est(2*i+1) = fv*state_est(3+3*i+2)/state_est(3+3*i) + v0; 
	}
}

void EKFobst::calculate_F(MatrixXf& F){
	F = MatrixXf::Identity(state.size(), state.size());
	int num_obstacles = (state.size() - 3)/3;
	for (int i = 0; i < num_obstacles; i++){
		F.block(3+3*i,0,3,3) = dt*Matrix3f::Identity(3,3);
	}
}

void EKFobst::calculate_H(MatrixXf& H){
	int num_obstacles = (state.size())/3;
	std::cout << "no: " << num_obstacles << std::endl; 
	H.setZero(num_obstacles*2, 3+num_obstacles*3);
	std::cout << "con't set zero" << std::endl; 
	// obstacle landmarks 
	for (int i = 0; i < num_obstacles; i++){
		H(2*i,3+3*i) = -fu*state(3+3*i+1)/(state(3+3*i)*state(3+3*i));
		H(2*i+1,3+3*i) = -fv*state(3+3*i+2)/(state(3+3*i)*state(3+3*i));
		H(2*i,3+3*i+1) = fu/state(3+3*i);
		H(2*i+1,3+3*i+2) = fv/state(3+3*i);
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
	// also update Q
	Q = process_noise*MatrixXf::Identity(state.size(), state.size());
	Q.bottomRightCorner(state.size()-3, state.size()-3) = dt*MatrixXf::Identity(state.size()-3, state.size()-3);
	int num_obstacles = (state.size() - 3)/3;
}	

void EKFobst::delete_landmark(int indx){
	int curr_num_obstacles = state.size()/3;
	MatrixXf new_covar;
	new_covar.setZero(state.size()-3, state.size()-3); 
	new_covar.topLeftCorner(3+3*indx, 3+3*indx) = covariance.topLeftCorner(3+3*indx,3+3*indx); 
	VectorXf new_state;
	new_state.setZero(state.size()-3); 
	new_state.head(3+3*indx) = state.head(3+3*indx); 
	int bc_size = new_state.size() - (3+3*indx);
	new_covar.bottomRightCorner(bc_size, bc_size) = covariance.bottomRightCorner(bc_size, bc_size);
	new_state.tail(bc_size) = state.tail(bc_size); 
	// assign to state 
	state = new_state; 
	covariance = new_covar;
	// also update Q
	Q = process_noise*MatrixXf::Identity(state.size(), state.size());
	Q.bottomRightCorner(state.size()-3, state.size()-3) = dt*MatrixXf::Identity(state.size()-3, state.size()-3);
	int num_obstacles = (state.size() - 3)/3;
}

void EKFobst::prediction(VectorXf& state_est, MatrixXf& P_est){
	calculate_state_est(state_est); // with motion update (prediction)
	// now calculate new estimated covariance P_est
	MatrixXf F; 
	calculate_F(F); // motion update matrix
	P_est = F*covariance*F.transpose() + Q;
}

void EKFobst::update(VectorXf measurement, VectorXf state_est, MatrixXf P_est, MatrixXf R){
	if (measurement.size() > 0){
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
	}else{
		state = state_est; 
		covariance = P_est; 
	}
}

void EKFobst::get_state(VectorXf& x){
	x = state;
}
void EKFobst::get_covariance(MatrixXf& P){
	P = covariance;
}