%% EKF Beacon Navigation - Car on a Straight 4m-Wide Road
%  State: z = [px; py; vx; vy; ax; ay]  (6-D)
%  Road:  runs along the x-axis, width 4 m (y in [-2, +2])
%  Beacons: placed in pairs along the two road edges
%  Dynamics: constant acceleration (nearly) - linear
%  Measurement: noisy Euclidean distances to each beacon - nonlinear, so EKF

clear; clc; close all;
rng(37);

% Sanity check: confirm casadi is on the path
if isempty(which('casadi.SX'))
    error(['casadi not found on MATLAB path. ', ...
           'Run: addpath(''/path/to/your/casadi/folder'')']);
end

%% Problem data
% Beacons along both edges of the 4 m road, spaced every 20 m over 60 m
%anchors = [  0,  2;   0, -2;
%            20,  2;  20, -2;
%            40,  2;  40, -2;
%            60,  2;  60, -2];

anchors = [  0,  -3;   
             50,  3;  
             100,  -3];

ny = size(anchors, 1);   % 8 measurements per step
nx = 6;                  % state dimension

%Q  = 0.01 * eye(2);      % process noise covariance (on acceleration)
Q = diag([0.01, 0.00001]);   % longitudinal noise unchanged, lateral 100x smaller

R  = 0.04 * eye(ny);     % measurement noise covariance (sigma ~ 0.2 m)
dt = 0.2;                % sampling interval (s)

% Constant-acceleration dynamics: acceleration persists with small noise
% (replaces the notebook's rotating Phi with identity)
Phi = eye(2);

A = [eye(2),      dt*eye(2),   zeros(2,2);
     zeros(2,2),  eye(2),      dt*eye(2);
     zeros(2,4),               Phi];

% Lifted process noise covariance (noise only enters acceleration)
G    = [zeros(4,2); eye(2)];
Qbar = G * Q * G';

%% System dynamics (linear)
f = @(z, w) A*z + G*w;

%% Measurement function and its Jacobian via casadi
z_ = casadi.SX.sym('z_', nx);
v_ = casadi.SX.sym('v_', ny);
r_ = z_(1:2);

h_expr = casadi.SX.zeros(ny, 1);
for i = 1:ny
    h_expr(i) = norm_2(r_ - anchors(i, :)') + v_(i);
end

jhx_expr = jacobian(h_expr, z_);

h   = casadi.Function('h',   {z_, v_}, {h_expr});
jhx = casadi.Function('jhx', {z_, v_}, {jhx_expr});

%% Noise generators (Cholesky instead of mvnrnd, no toolbox needed)
L_R = chol(R, 'lower');
L_Q = chol(Q, 'lower');

%% Simulation setup
N = 80;

% True initial state: at origin, cruising at 5 m/s along +x
x_t = [0; 0; 5; 0; 0; 0];

% Filter's initial prior: origin, zero velocity, zero acceleration, loose cov
x_t_pred     = zeros(nx, 1);
sigma_t_pred = 10 * eye(nx);

% Storage
x_cache          = zeros(N,   nx);
x_meas_cache     = zeros(N-1, nx);
sigma_meas_cache = zeros(nx, nx, N-1);

x_cache(1, :) = x_t';

%% Main EKF loop
for t = 1:N-1
    % i. Obtain measurement (simulator)
    v_t = L_R * randn(ny, 1);
    y_t = full(h(x_t, v_t));

    % ii. Measurement update (filter)
    [sigma_t_meas, x_t_meas] = measurement_update(...
        sigma_t_pred, x_t_pred, y_t, h, jhx, R, ny);
    x_meas_cache(t, :)        = x_t_meas';
    sigma_meas_cache(:, :, t) = sigma_t_meas;

    % iii. Time update (filter)
    [sigma_t_pred, x_t_pred] = time_update(...
        sigma_t_meas, x_t_meas, A, Qbar, f);

    % iv. Advance true dynamics (simulator)
    w_t = L_Q * randn(2, 1);
    x_t = f(x_t, w_t);
    x_cache(t+1, :) = x_t';
end

%% Plot trajectory with road edges and uncertainty ellipses
figure('Position', [100 100 1100 300]);
hold on; grid on;

% Road edges
x_road = [min(anchors(:,1))-5, max(anchors(:,1))+5];
plot(x_road, [ 2,  2], 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
plot(x_road, [-2, -2], 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
plot(x_road, [ 0,  0], 'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');

% Filter estimate and truth
plot(x_meas_cache(:,1), x_meas_cache(:,2), 'b-', 'LineWidth', 1.4, ...
    'DisplayName', 'EKF estimate');
plot(x_cache(:,1), x_cache(:,2), 'r-', 'LineWidth', 1.4, ...
    'DisplayName', 'True trajectory');

% Beacons
for i = 1:ny
    if i == 1
        plot(anchors(i,1), anchors(i,2), 'kx', 'MarkerSize', 10, ...
            'LineWidth', 2, 'DisplayName', 'Beacons');
    else
        plot(anchors(i,1), anchors(i,2), 'kx', 'MarkerSize', 10, ...
            'LineWidth', 2, 'HandleVisibility', 'off');
    end
end

% Uncertainty ellipses every 4 steps
for t = 1:4:N-1
    mu  = x_meas_cache(t, 1:2)';
    cov = sigma_meas_cache(1:2, 1:2, t);
    plot_2d_normal_contour(mu, cov, 0.4);
end

legend('Location', 'northeast');
xlabel('x-coordinate (m)  --  along road');
ylabel('y-coordinate (m)  --  across road');
title('EKF: car on a 4 m road tracked by edge beacons');
%axis equal;
ylim([-4, 4]);

%% Optional diagnostic plot: estimation errors over time
figure('Position', [100 450 900 500]);
time = (0:N-2) * dt;

subplot(3,1,1);
plot(time, x_cache(1:N-1,1) - x_meas_cache(:,1), 'b', ...
     time, x_cache(1:N-1,2) - x_meas_cache(:,2), 'r');
legend('x-error', 'y-error'); ylabel('Position error (m)'); grid on;

subplot(3,1,2);
plot(time, x_cache(1:N-1,3) - x_meas_cache(:,3), 'b', ...
     time, x_cache(1:N-1,4) - x_meas_cache(:,4), 'r');
legend('vx-error', 'vy-error'); ylabel('Velocity error (m/s)'); grid on;

subplot(3,1,3);
plot(time, x_cache(1:N-1,5) - x_meas_cache(:,5), 'b', ...
     time, x_cache(1:N-1,6) - x_meas_cache(:,6), 'r');
legend('ax-error', 'ay-error'); ylabel('Acceleration error (m/s^2)');
xlabel('Time (s)'); grid on;

sgtitle('Estimation errors (truth - estimate)');

%% ----- Local functions -----

function [sigma_meas, x_meas] = measurement_update(...
        sigma_pred, x_pred, y, h, jhx, R, ny)

    % 1. Linearise h around the current estimate
    C = full(jhx(x_pred, zeros(ny, 1)));

    % 2. Predict what the sensor WOULD read if x_pred were correct
    y_predicted = full(h(x_pred, zeros(ny, 1)));

    % 3. Surprise (innovation)
    innovation = y - y_predicted;

    % 4. Combined uncertainty: from state (C*sigma*C') and from sensor (R)
    prediction_covariance = C * sigma_pred * C' + R;

    % 5. Kalman gain: how strongly should we trust this measurement?
    kalman_gain = sigma_pred * C' / prediction_covariance;

    % 6. Corrected estimate
    x_meas = x_pred + kalman_gain * innovation;

    % 7. Shrunk covariance (we just learned something)
    sigma_meas = sigma_pred - kalman_gain * C * sigma_pred;
end

function [sigma_pred, x_pred] = time_update(sigma_meas, x_meas, A, Qbar, f)
    % Predict next state with zero process noise (noise has mean zero)
    x_pred     = f(x_meas, zeros(2, 1));
    % Inflate covariance: old uncertainty pushed forward + fresh process noise
    sigma_pred = A * sigma_meas * A' + Qbar;
end

function plot_2d_normal_contour(mu, cov, alpha)
    % 1-sigma elliptical contour of the 2D Gaussian N(mu, cov)
    theta   = linspace(0, 2*pi, 80);
    [V, D]  = eig(cov);
    circle  = [cos(theta); sin(theta)];
    ellipse = V * sqrt(D) * circle;
    plot(mu(1) + ellipse(1,:), mu(2) + ellipse(2,:), ...
        'Color', [0 0 0 alpha], 'HandleVisibility', 'off');
end