function sigma_points = generate_sigma_points(predicted_state,predicted_covariance)
n = length(predicted_state);

square_root_covariance = sqrtm(predicted_covariance);

sigma_points = zeros(n, 2*n+1);
sigma_points(:,1) = predicted_state;

    for i = 1:n
        sigma_points(:,i+1)   = predicted_state + sqrt(n/(1-0.1))*square_root_covariance(:,i);
        sigma_points(:,i+n+1) = predicted_state - sqrt(n/(1-0.1))*square_root_covariance(:,i);
    end
end