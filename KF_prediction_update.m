function [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(prev_state,prev_covariance,A,B,C,Q)
    predicted_state= A*prev_state;
    predicted_covariance = A*prev_covariance*A'+B*Q*B';
    predicted_measurement = C*predicted_state;
end