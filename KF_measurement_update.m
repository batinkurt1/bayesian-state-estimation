function [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,C,H,R)
    S = C*predicted_covariance*C' + H*R*H';
    K = predicted_covariance*C'/S;
    estimated_state = predicted_state + K*(actual_measurement - predicted_measurement);
    estimated_covariance = predicted_covariance - K*S*K';
    estimated_covariance = 1/2*(estimated_covariance+estimated_covariance');
end