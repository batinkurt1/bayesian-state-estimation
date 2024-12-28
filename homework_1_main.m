clearvars; clc; 
close all;
%Loading the true data belonging to the target 
load("trueTarget.mat");

time_steps = length(trueTarget(1, :));
x_true = trueTarget(2, :);
y_true = trueTarget(3, :);

figure;
subplot(2,2,1);
plot(1:time_steps,x_true,LineWidth=1.5,Color="#A2142F");
title("True x-position of the Target vs. Time");
xlabel("k");
xticks(0:10:150);
ylabel("x position");
ylim([500,3000]);
legend("x position");
grid on;

subplot(2,2,3);
plot(1:time_steps,y_true,LineWidth=1.5,Color="#D95319");
title("True y-position of the Target vs. Time");

xlabel("k");
xticks(0:10:150);
ylabel("y position");
ylim([0,2500]);
grid on;
legend("y position");


subplot(2,2,[2,4]);
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
title("True Target Trajectory");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory");
grid on;

%sampling period
T=1;

% State transition matrix A
A = [eye(2),T*eye(2);
    zeros(2,2),eye(2)];

% Process Noise matrix B
B = [T^2/2*eye(2);
    T*eye(2)];

Q=eye(2);

noisy_measurements = zeros(2,time_steps);

sigma_x = 100;  % Standard deviation for x measurement noise 
mu_x = 0;       % Mean for x measurement noise 
sigma_y = 100;  % Standard deviation for y measurement noise 
mu_y = 0;       % Mean for y measurement noise 

R = diag([sigma_x^2,sigma_y^2]);

measurement_noise_mu = [mu_x; mu_y];

for k=1:time_steps
    noisy_measurements(:,k) = generate_measurements(trueTarget(2:3,k),measurement_noise_mu,R);
end

x_noisy = noisy_measurements(1,:);
y_noisy = noisy_measurements(2,:);

% Plot the True Trajectory and Noisy Measurements
figure;
plot(x_true, y_true, LineWidth=1.5, Color="#77AC30"); % True trajectory in green
hold on;
plot(x_noisy, y_noisy, 'r.'); % Noisy measurements in red dots
legend('True Trajectory', 'Noisy Measurements');
xlabel('x position');
ylabel('y position');
title('True Target Trajectory vs. Noisy Measurements');
grid on;

C = [eye(2),zeros(2)];
H = eye(2);

% Initial state mean
x0_bar = [1000; 1000; 0; 0];

% Initial state covariance P0
P0 = diag([100^2, 100^2, 10^2, 10^2]);

x0 = mvnrnd(x0_bar,P0)';

%Implement Kalman Filter
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(prev_state,prev_covariance,A,B,C,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        actual_measurement = noisy_measurements(:,k);
        [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,C,H,R);
        estimated_states(:,k) = estimated_state;
end

figure;

plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
title("True Target Trajectory vs. Conventional Kalman Filter");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Conventional Kalman Filter");
grid on;

estimation_error = sqrt((x_true-estimated_states(1,:)).^2+(y_true-estimated_states(2,:)).^2);
prediction_error = sqrt((x_true-predicted_states(1,:)).^2+(y_true-predicted_states(2,:)).^2);

figure;
plot(1:time_steps,estimation_error,LineWidth=1.5,Color="#77AC30");
hold on;
plot(1:time_steps,prediction_error,LineWidth=1.5,Color="#D95319");
grid on;
legend("Estimation Error","Prediction Error");
xlabel("Time Steps");
ylabel("Position Error (m)");
title("Estimation Error and Prediction Error vs. Time Steps");

rms_estimation = sqrt(1/time_steps*(sum(estimation_error.^2)));
rms_prediction = sqrt(1/time_steps*(sum(prediction_error.^2)));

fprintf("Root Mean Square Error of Estimated Position: %0.5g \n",rms_estimation);
fprintf("Root Mean Square Error of Predicted Position: %0.5g \n",rms_prediction);

%increase R to 100*R.

R = 100*diag([sigma_x^2,sigma_y^2]);

%Implement Kalman Filter
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(prev_state,prev_covariance,A,B,C,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        actual_measurement = noisy_measurements(:,k);
        [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,C,H,R);
        estimated_states(:,k) = estimated_state;
end

figure;
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
plot(x_noisy, y_noisy, 'r.'); % Noisy measurements in red dots
title("True Target Trajectory vs. Conventional Kalman Filter under Model Mismatch (100R)");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Conventional Kalman Filter under Model Mismatch","Noisy Measurements");
grid on;

%Measurment Noise is much higher than it is supposed to be. The filter
%disregards the measurements and trusts its predictions more.

%decrease R to 1/100*R.
R = 1/100*diag([sigma_x^2,sigma_y^2]);

for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(prev_state,prev_covariance,A,B,C,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        actual_measurement = noisy_measurements(:,k);
        [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,C,H,R);
        estimated_states(:,k) = estimated_state;
end

figure;
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
plot(x_noisy, y_noisy, 'r.'); % Noisy measurements in red dots
title("True Target Trajectory vs. Conventional Kalman Filter under Model Mismatch (R/100)");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Conventional Kalman Filter under Model Mismatch","Noisy Measurements");
grid on;

%Measurment Noise is much lower than it is supposed to be. The filter
%does not trust its predictions and cares too much about the measurements.

%increase Q to 100*Q. set R as initial value.

R = diag([sigma_x^2,sigma_y^2]);
Q=100*eye(2);

%Implement Kalman Filter
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(prev_state,prev_covariance,A,B,C,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        actual_measurement = noisy_measurements(:,k);
        [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,C,H,R);
        estimated_states(:,k) = estimated_state;
end

figure;

plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
plot(x_noisy, y_noisy, 'r.'); % Noisy measurements in red dots
title("True Target Trajectory vs. Conventional Kalman Filter under Model Mismatch (100Q)");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Conventional Kalman Filter under Model Mismatch","Noisy Measurements");
grid on;

%Process Noise is much higher than it is supposed to be. The filter
%does not trust its predictions and cares too much about the measurements.



%decrease Q to 1/100*Q. set R as initial value.

R = diag([sigma_x^2,sigma_y^2]);
Q=1/100*eye(2);

%Implement Kalman Filter
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = KF_prediction_update(prev_state,prev_covariance,A,B,C,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        actual_measurement = noisy_measurements(:,k);
        [estimated_state,estimated_covariance] = KF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,C,H,R);
        estimated_states(:,k) = estimated_state;
end

figure;

plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
plot(x_noisy, y_noisy, 'r.'); % Noisy measurements in red dots
title("True Target Trajectory vs. Conventional Kalman Filter under Model Mismatch (Q/100)");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Conventional Kalman Filter under Model Mismatch","Noisy Measurements");
grid on;
%Process Noise is much lower than it is supposed to be. The filter
%disregards the measurements and trusts its predictions more.

noisy_measurements = zeros(2,time_steps);

sigma_r = 100;
mu_r = 0;
sigma_theta = 5*pi/180;
mu_theta = 0;

R = diag([sigma_r^2,sigma_theta^2]);

measurement_noise_mu = [mu_r; mu_theta];

r_true = sqrt(trueTarget(2,:).^2+trueTarget(3,:).^2);
theta_true = atan2(trueTarget(3,:),trueTarget(2,:));



for k=1:time_steps
    noisy_measurements(:,k) = generate_measurements([r_true(k);theta_true(k)],measurement_noise_mu,R);
end

r_noisy = noisy_measurements(1,:);
theta_noisy = noisy_measurements(2,:);


figure;
subplot(2,2,1)
plot(1:time_steps, r_true, LineWidth=1.5, Color="#77AC30"); % True range in green
hold on;
plot(1:time_steps, r_noisy, 'r.'); % Noisy measurements in red dots
legend('True Target Range', 'Noisy Range');
xlabel('Time steps');
ylabel('Range (m)');
title('True Target Range vs. Noisy Range');
grid on;
hold off;

subplot(2,2,3)
plot(1:time_steps, theta_true, LineWidth=1.5, Color="#77AC30"); % True range in green
hold on;
plot(1:time_steps, theta_noisy, 'r.'); % Noisy measurements in red dots
legend('True Target Bearing', 'Noisy Bearing');
xlabel('Time steps');
ylabel('Bearing (degrees)');
title('True Target Bearing vs. Noisy Bearing');
grid on;
hold off;

subplot(2,2,[2,4]);

plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(r_noisy.*cos(theta_noisy),r_noisy.*sin(theta_noisy), 'r.');
title("True Target Trajectory vs. Noisy Target Trajectory");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Noisy Target Trajectory");
grid on;


h = @(x)[sqrt(x(1,:).^2+x(2,:).^2);atan2(x(2,:),x(1,:))]; 


R = diag([sigma_r^2,sigma_theta^2]);
Q=eye(2);


% EKF Implementation
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = EKF_prediction_update(prev_state,prev_covariance,A,B,h,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;

        actual_measurement = noisy_measurements(:,k);
        [estimated_state,estimated_covariance] = EKF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,h,R);
        estimated_states(:,k) = estimated_state;
end


figure;
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
title("True Target Trajectory vs. Extended Kalman Filter");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Extended Kalman Filter");
grid on;

estimation_error = sqrt((x_true-estimated_states(1,:)).^2+(y_true-estimated_states(2,:)).^2);


rms_estimation = sqrt(1/time_steps*(sum(estimation_error.^2)));


fprintf("Root Mean Square Error of Estimated Position in EKF: %0.5g \n",rms_estimation);


%IEKF Implementation 
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = EKF_prediction_update(prev_state,prev_covariance,A,B,h,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        
        actual_measurement = noisy_measurements(:,k);
        for j = 1:10
        if j == 1 
            jacobian_location = predicted_state;
        else
            jacobian_location = estimated_state;
        end
        [estimated_state,estimated_covariance] = IEKF_measurement_update(predicted_state,predicted_covariance,predicted_measurement,actual_measurement,h,R,jacobian_location);
        end 
        estimated_states(:,k) = estimated_state;
end

figure;
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
title("True Target Trajectory vs. Iterated Extended Kalman Filter");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Iterated Extended Kalman Filter");
grid on;

estimation_error = sqrt((x_true-estimated_states(1,:)).^2+(y_true-estimated_states(2,:)).^2);


rms_estimation = sqrt(1/time_steps*(sum(estimation_error.^2)));


fprintf("Root Mean Square Error of Estimated Position in IEKF: %0.5g \n",rms_estimation);

%UKF Implementation
for k=1:time_steps
    if k==1
       estimated_states = zeros(4,time_steps);
       predicted_states = zeros(4,time_steps);
       predicted_measurements = zeros(2,time_steps);
       prev_state = x0_bar;
       estimated_covariance = P0;
    else
        prev_state = estimated_states(:,k-1);
    end
        prev_covariance = estimated_covariance;
        [predicted_state,predicted_covariance,predicted_measurement] = EKF_prediction_update(prev_state,prev_covariance,A,B,h,Q);
        predicted_states(:,k) = predicted_state;
        predicted_measurements(:,k) = predicted_measurement;
        
        actual_measurement = noisy_measurements(:,k);

        sigma_points = generate_sigma_points(predicted_state,predicted_covariance);
        transformed_sigma_points = h(sigma_points);
        W = calculate_UT_weights(length(predicted_state));
        [estimated_state,estimated_covariance] = UKF_measurement_update(sigma_points,transformed_sigma_points,W,W,R,actual_measurement,predicted_state,predicted_covariance);
        estimated_states(:,k) = estimated_state;
end

figure;
plot(x_true,y_true,LineWidth=1.5,Color="#77AC30");
hold on;
plot(estimated_states(1,:),estimated_states(2,:),LineWidth=1.5,Color="#D95319")
title("True Target Trajectory vs. Unscented Kalman Filter");
ylabel("y position");
ylim([0,2500]);
xlabel("x-position");
xlim([500,3000]);
legend("True Target Trajectory","Unscented Kalman Filter");
grid on;

estimation_error = sqrt((x_true-estimated_states(1,:)).^2+(y_true-estimated_states(2,:)).^2);


rms_estimation = sqrt(1/time_steps*(sum(estimation_error.^2)));


fprintf("Root Mean Square Error of Estimated Position in UKF: %0.5g \n",rms_estimation);
