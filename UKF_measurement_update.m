function [estimated_state,estimated_covariance] = UKF_measurement_update(sigma_points,transformed_sigma_points,Wm,Wc,R,actual_measurement,predicted_state,predicted_covariance)
n = size(transformed_sigma_points,1);
predicted_measurement_covariance = zeros(n,n);
cross_covariance = zeros(size(predicted_state,1),n);
predicted_measurement = transformed_sigma_points*Wm;

for j=1:(2*size(predicted_state,1)+1)
    y = transformed_sigma_points(:,j) - predicted_measurement;
    predicted_measurement_covariance = predicted_measurement_covariance + Wc(j) * (y * y');
    cross_covariance = cross_covariance+Wc(j)*(sigma_points(:,j)-predicted_state)*y';
end
predicted_measurement_covariance = predicted_measurement_covariance + R;

K = cross_covariance/predicted_measurement_covariance;
estimated_state = predicted_state + K*(actual_measurement - predicted_measurement);
estimated_covariance = predicted_covariance - K*predicted_measurement_covariance*K';


