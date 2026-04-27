%% EKF Beacon Navigation Example (MATLAB + casadi)
% State: z = [position(2); velocity(2); acceleration(2)], 6-D total
% Measurement: noisy distances to 3 fixed beacons (nonlinear in position)

clear; clc; close all;
import casadi.*
rng(37);

%% Problem data
Phi = [0.50,  0.87;
      -0.87,  0.48];

%anchors = [  0, -3;
%             10,  3;
%             20, -3];

anchors = [3, 2;
           2, -3; 
          -5, -3];

ny = size(anchors, 1);   % number of beacons = 3
nx = 6;                  % state dimension

Q  = 0.02 * eye(2);      % process noise covariance (acceleration only)
R  = 0.05 * eye(ny);     % measurement noise covariance
dt = 0.2;                % sampling time

% Full state-transition matrix (linear dynamics)
A = [eye(2),      dt*eye(2),   zeros(2,2);
     zeros(2,2),  eye(2),      dt*eye(2);
     zeros(2,4),               Phi];

% Lifted process noise covariance (noise only enters acceleration)
G    = [zeros(4,2); eye(2)];
Qbar = G * Q * G';

%% System dynamics (linear, no casadi needed)
f = @(z, w) A*z + G*w;

%% Measurement function and its Jacobians via casadi
z_ = SX.sym('z_', nx);
v_ = SX.sym('v_', ny);
r_ = z_(1:2);

% Build h symbolically
h_expr = SX.zeros(ny, 1);
for i = 1:ny
    h_expr(i) = norm_2(r_ - anchors(i, :)') + v_(i);
end

% Symbolic Jacobians
jhx_expr = jacobian(h_expr, z_);
jhv_expr = jacobian(h_expr, v_);

% Wrap as callable Function objects
h   = Function('h',   {z_, v_}, {h_expr});
jhx = Function('jhx', {z_, v_}, {jhx_expr});
jhv = Function('jhv', {z_, v_}, {jhv_expr});  % unused; noise is additive

%% Simulation setup
N = 100;

x_t          = [0.5; 0.1; 1; 0; 0; 0];   % true initial state
x_t_pred     = zeros(nx, 1);              % filter's prior mean
sigma_t_pred = 10 * eye(nx);              % filter's prior covariance

% Storage
x_cache          = zeros(N,   nx);
x_meas_cache     = zeros(N-1, nx);
sigma_meas_cache = zeros(nx, nx, N-1);

x_cache(1, :) = x_t';

%% Main EKF loop
for t = 1:N-1
    % i. Obtain measurement (the "real world" producing y_t)
    v_t = chol(R, 'lower') * randn(ny, 1);
    y_t = full(h(x_t, v_t));

    % ii. Measurement update
    [sigma_t_meas, x_t_meas] = measurement_update(...
        sigma_t_pred, x_t_pred, y_t, h, jhx, R, ny);
    x_meas_cache(t, :)        = x_t_meas';
    sigma_meas_cache(:, :, t) = sigma_t_meas;

    % iii. Time update
    [sigma_t_pred, x_t_pred] = time_update(...
        sigma_t_meas, x_t_meas, A, Qbar, f);

    % iv. Advance true dynamics
    w_t = chol(Q, 'lower') * randn(2, 1);
    x_t = f(x_t, w_t);
    x_cache(t+1, :) = x_t';
end

%% Plot
figure('Position', [100 100 600 400]);
hold on; grid on;

plot(x_meas_cache(:,1), x_meas_cache(:,2), 'b-', 'LineWidth', 1.2, ...
    'DisplayName', 'Measurement update');
plot(x_cache(:,1), x_cache(:,2), 'r-', 'LineWidth', 1.2, ...
    'DisplayName', 'Actual state');

for i = 1:ny
    if i == 1
        plot(anchors(i,1), anchors(i,2), 'kx', 'MarkerSize', 10, ...
            'LineWidth', 2, 'DisplayName', 'Anchors');
    else
        plot(anchors(i,1), anchors(i,2), 'kx', 'MarkerSize', 10, ...
            'LineWidth', 2, 'HandleVisibility', 'off');
    end
end

% Uncertainty ellipses every 3rd step
for t = 1:3:N-1
    mu  = x_meas_cache(t, 1:2)';
    cov = sigma_meas_cache(1:2, 1:2, t);
    plot_2d_normal_contour(mu, cov, 0.3);
end

legend('Location', 'best');
xlabel('x-coordinate');
ylabel('y-coordinate');
title('EKF: beacon navigation (casadi)');
axis equal;

%% ----- Local functions -----

function [sigma_meas, x_meas] = measurement_update(...
        sigma_pred, x_pred, y, h, jhx, R, ny)
    C          = full(jhx(x_pred, zeros(ny, 1)));
    Z          = C * sigma_pred * C' + R;
    innovation = y - full(h(x_pred, zeros(ny, 1)));
    x_meas     = x_pred + sigma_pred * C' * (Z \ innovation);
    sigma_meas = sigma_pred - sigma_pred * C' * (Z \ (C * sigma_pred));
end

function [sigma_pred, x_pred] = time_update(sigma_meas, x_meas, A, Qbar, f)
    x_pred     = f(x_meas, zeros(2, 1));
    sigma_pred = A * sigma_meas * A' + Qbar;
end

function plot_2d_normal_contour(mu, cov, alpha)
    theta   = linspace(0, 2*pi, 80);
    [V, D]  = eig(cov);
    circle  = [cos(theta); sin(theta)];
    ellipse = V * sqrt(D) * circle;
    plot(mu(1) + ellipse(1,:), mu(2) + ellipse(2,:), ...
        'Color', [0 0 0 alpha], 'HandleVisibility', 'off');
end