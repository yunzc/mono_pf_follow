#include "EKF.h" 

void EKF::load_initial(){
	std::cout << "not yet implemented" << std::endl; 
}

void EKF::calculate_Rot(Matrix3f& R){
	// rotation matrix according to the rotation rate
	float phi = state(5)*dt;
	float theta = state(4)*dt; 
	float psi = state(3)*dt;
	R(0,0) = cos(phi)*cos(theta); 
	R(1,0) = sin(phi)*cos(theta); 
	R(2,0) = -sin(theta);
	R(0,1) = cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi);
	R(1,1) = sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi);
	R(2,1) = cos(theta)*sin(psi);
	R(0,2) = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi);
	R(1,2) = sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi); 
	R(2,2) = cos(theta)*cos(psi);
}

void EKF::calculate_dRdphi(Matrix3f& mat){
	// rotation matrix according to the rotation rate
	float phi = state(5)*dt;
	float theta = state(4)*dt; 
	float psi = state(3)*dt;
	R(0,0) = -sin(phi)*cos(theta); 
	R(1,0) = cos(phi)*cos(theta); 
	R(2,0) = 0;
	R(0,1) = -sin(phi)*sin(theta)*sin(psi) - cos(phi)*cos(psi);
	R(1,1) = cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi);
	R(2,1) = 0;
	R(0,2) = -sin(phi)*sin(theta)*cos(psi) + cos(phi)*sin(psi);
	R(1,2) = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi); 
	R(2,2) = 0;
}

void EKF::calculate_dRdthet(Matrix3f& mat){
	// rotation matrix according to the rotation rate
	float phi = state(5)*dt;
	float theta = state(4)*dt; 
	float psi = state(3)*dt;
	R(0,0) = -cos(phi)*sin(theta); 
	R(1,0) = -sin(phi)*sin(theta); 
	R(2,0) = -cos(theta);
	R(0,1) = cos(phi)*cos(theta)*sin(psi);
	R(1,1) = sin(phi)*cos(theta)*sin(psi);
	R(2,1) = -sin(theta)*sin(psi);
	R(0,2) = cos(phi)*cos(theta)*cos(psi);
	R(1,2) = sin(phi)*cos(theta)*cos(psi); 
	R(2,2) = -sin(theta)*cos(psi);
}

void EKF::calculate_dRdpsi(Matrix3f& mat){
	// rotation matrix according to the rotation rate
	float phi = state(5)*dt;
	float theta = state(4)*dt; 
	float psi = state(3)*dt;
	R(0,0) = 0; 
	R(1,0) = 0; 
	R(2,0) = 0;
	R(0,1) = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi);
	R(1,1) = sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi);
	R(2,1) = cos(theta)*cos(psi);
	R(0,2) = -cos(phi)*sin(theta)*sin(psi) + sin(phi)*cos(psi);
	R(1,2) = -sin(phi)*sin(theta)*sin(psi) - cos(phi)*cos(psi); 
	R(2,2) = -cos(theta)*sin(psi);
}

void EKF::calculate_state_est(VectorXf& state_est){
	// state estimation according to motion model 
	Matrix3f R; 
	calculate_Rot(R); 
	VectorXf newstate(state.size());
	newstate.segment(0,6) = state.segment(0,6);
	VectorXf target_state(6); 
	target_state(0) = state(6) - state(0)*dt + state(9)*dt;
	target_state(1) = state(7) - state(1)*dt + state(10)*dt; 
	target_state(2) = state(8) - state(2)*dt + state(11)*dt; 
	target_state(3) = state(9);
	target_state(4) = state(10);
	target_state(5) = state(11);
	newstate.segment(6,3) = R*target_state.segment(0,3);
	newstate.segment(9,3) = R*target_state.segment(3,3);
	for (int i = 12; i < state.size(); i += 3){
		Vector3f obstacle_state;
		obstacle_state(0) = state(i) - state(0)*dt; 
		obstacle_state(1) = state(i+1) - state(1)*dt; 
		obstacle_state(2) = state(i+2) - state(2)*dt; 
		newstate.segment(i,3) = R*obstacle_state; 
	}
}
void EKF::calculate_meas_est(VectorXf state_est, VectorXf& meas_est){
	// calculate measurement from state estimate 
	int num_obstacles = (state_est.size() - 12)/3;
	meas_est.setZero(num_obstacles + 4);
	meas_est(0) = f*(state_est(7) - w/2)/state_est(6) + u0; 
	meas_est(1) = f*(state_est(8) - h/2)/state_est(6) + v0; 
	meas_est(2) = f*(state_est(7) + w/2)/state_est(6) + u0;
	meas_est(3) = f*(state_est(8) + h/2)/state_est(6) + v0; 
	for (int i = 0; i < num_obstacles; i++){
		meas_est(4+2*i) = f*state_est(12+3*i+1)/state_est(12+3*i) + u0; 
		meas_est(4+2*i+1) = f*state_est(12+3*i+2)/state_est(12+3*i) + v0; 
	}
}

void EKF::calculate_F(MatrixXf& F){
	F.setZero(state.size(), state.size());
	F.topLeftCorner(6,6) = MatrixXf::Identity(6,6); 
	Matrix3f R; 
	calculate_Rot(R);
	// [dft/dvx, dft/dvy, dft/dvz]
	F.block(6,0,3,3) = -dt*R;
	F.block(9,0,3,3) = R;
	VectorXf target_state(6); 
	target_state(0) = state(6) - state(0)*dt + state(9)*dt;
	target_state(1) = state(7) - state(1)*dt + state(10)*dt; 
	target_state(2) = state(8) - state(2)*dt + state(11)*dt; 
	target_state(3) = state(9);
	target_state(4) = state(10);
	target_state(5) = state(11);
	// dft/dwx
	Matrix3f dRdpsi; 
	calculate_dRdpsi(dRdpsi);
	F.block(6,3,3,1) = dRdpsi*target_state.segment(0,3);
	F.block(9,3,3,1) = dRdpsi*target_state.segment(3,3);
	// dft/dwy 
	Matrix3f dRdthet;
	calculate_dRdthet(dRdthet);
	F.block(6,4,3,1) = dRdthet*target_state.segment(0,3);
	F.block(9,4,3,1) = dRdthet*target_state.segment(3,3);
	//dft/dwz 
	Matrix3f dRdphi; 
	calculate_dRdphi(dRdphi);
	F.block(6,5,3,1) = dRdphi*target_state.segment(0,3);
	F.block(6,5,3,1) = dRdphi*target_state.segment(3,3);
	// [dft/dx, dft/dy. dft/dz] 
	F.block(6,6,3,3) = R; 
	// zeros for F.block(9,6,3,3) 
	// [dft/dxdot, dft/dydot, dft/dzdot] are zeros 
	F.block(6,9,3,3) = R*dt;
	F.block(9,9,3,3) = R; 
	int num_obstacles = (state.size() - 12)/3;
	for (int i = 0; i < num_obstacles; i++){
		// [dfy/dvx, dfy/dvy, dfy/dvz]
		F.block(12+3*i,0,3,3) = -dt*R; 
		Vector3f obstacle_state;
		obstacle_state(0) = state(12+3*i) - state(0)*dt; 
		obstacle_state(1) = state(12+3*i+1) - state(1)*dt; 
		obstacle_state(2) = state(12+3*i+2) - state(2)*dt; 
		//dfy/dwx
		F.block(12+3*i,3,3,1) = dRdpsi*obstacle_state;
		//dfy/dwy
		F.block(12+3*i,4,3,1) = dRdthet*obstacle_state;
		//dfy/dwz
		F.block(12+3*i,5,3,1) = dRdphi*obstacle_state;
		//[dfy/dxi, dfy/dyi, dfy/dzi]
		F.block(12+3*i,12+3*i,3,3) = R;
	}
}

void EKF::calculate_H(MatrixXf& H){
	int num_obstacles = (state.size() - 12)/3;
	H.setZero(4 + num_obstacles*2, 12 + num_obstacles*3);
	// first 6 cols all zeros
	// the target 
	H(0,6) = -f*(state(7) - w/2)/(state(6)*state(6));
	H(1,6) = -f*(state(8) - h/2)/(state(6)*state(6));
	H(2,6) = -f*(state(7) + w/2)/(state(6)*state(6));
	H(3,6) = -f*(state(8) + h/2)/(state(6)*state(6));
	H.block(0,7,2,2) = f/state(6)*MatrixXf::Identity(2,2);
	H.block(3,7,2,2) = f/state(6)*MatrixXf::Identity(2,2);
	// obstacle landmarks 
	for (int i = 0; i < num_obstacles; i++){
		H(4+2*i,12+3*i) = -f*state(12+3*i+1)/(state(12+3*i)*state(12+3*i));
		H(4+2*i+1,12+3*i) = -f*state(12+3*i+2)/(state(12+3*i)*state(12+3*i));
		H(4+2*i,12+3*i+1) = f/state(12+3*i);
		H(4+2*i,12+3*i+2) = f/state(12+3*i);
	}
}

void EKF::add_landmark(Vector3f pos, Matrix3f pos_covar){
	MatrixXf new_covar;
	new_covar.setZero(state.size()+3, state.size()+3); 
	new_covar.topLeftCorner(state.size(), state.size()) = covariance; 
	VectorXf new_state;
	new_state.setZero(state.size()+3); 
	new_state.head(state.size()) = state; 
	new_covar.bottomRightCorner(3,3) = pos_covar; 
	new_state.tail(3) = pos; 
}

void EKF::delete_landmark(int indx){
	int curr_num_obstacles = (state.size()-12)/3;
	MatrixXf new_covar;
	new_covar.setZero(state.size()-3, state.size()-3); 
	new_covar.topLeftCorner(12+3*indx, 12+3*indx) = covariance.topLeftCorner(12+3*indx, 12+3*indx); 
	VectorXf new_state;
	new_state.setZero(state.size()-3); 
	new_state.head(12+3*indx) = state.head(12+3*indx); 
	int bc_size = new_state.size() - (12+3*indx);
	new_covar.bottomRightCorner(bc_size, bc_size) = covariance.bottomRightCorner(bc_size, bc_size);
	new_state.tail(bc_size) = state.tail(bc_size); 
}

void EKF::prediction(VectorXf& state_est, MatrixXf& P_est){
	calculate_state_est(state_est); // with motion update (prediction)
	// now calculate new estimated covariance P_est
	MatrixXf F; 
	calculate_F(F); // motion update matrix
	P_est = F*covariance*F + Q; // TODO check matrix mult in eigen
}

void EKF::update(VectorXf measurement, VectorXf state_est, MatrixXf P_est){
	VectorXf h; 
	calculate_meas_est(state_est, h);
	VectorXf y = measurement - h; // difference between measurement and estimate
	MatrixXf H; 
	calculate_H(H);
	MatrixXf S = H*P_est*H.transpose() + R; // TODO check matrix transpose/mult in eigen
	MatrixXf K = P_est*H.transpose()*S.inverse(); // Kalman gain 
	state = state_est + K*y; // update state estimate in EKF 
	MatrixXf I = MatrixXf::Identity(P_est.rows(), P_est.cols());
	covariance = (I - K*H)*P_est;	// update covariance 
}

void EKF::get_state(VectorXf& x){
	x = state;
}
void EKF::get_covariance(MatrixXf& P){
	P = covariance;
}