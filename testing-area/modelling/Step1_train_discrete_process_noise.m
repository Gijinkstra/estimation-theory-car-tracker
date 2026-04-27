%% Train Cruise Controller — Discrete-Time with Process Noise
% State:        z = [x; v]
% Control law:  F_t = K * (z_ref - z)
% Design:       Pole placement for desired wn, zeta
% Simulation:   Discrete-time with additive Gaussian process noise

clear; clc; close all;
rng(1);  % Fix random seed for reproducibility

%% 1. Physical parameters
m     = 4e5;        % Train mass [kg]
b     = 5000;       % Linear drag coefficient [N*s/m]
mu    = 0.0015;     % Rolling friction coefficient [-]
g     = 9.81;       % Gravity [m/s^2]

%% 2. Continuous state-space model (Option 2: augmented input)
A = [0   1;
     0  -b/m];

B = [0       0;
     1/m   -mu*g];

%% 3. Controller design via pole placement (still in continuous time)
wn   = 0.2;         % Natural frequency [rad/s]
zeta = 1.0;         % Damping ratio

desired_poly  = [1, 2*zeta*wn, wn^2];
desired_poles = roots(desired_poly);

% Only the first column of B drives the control input
B_ctrl = B(:,1);
K = place(A, B_ctrl, desired_poles);

fprintf('State feedback gain K = [k1, k2]:\n');
disp(K);

%% 4. Discretize the plant
dt = 0.5;           % Sample time [s]

sys_c = ss(A, B, eye(2), 0);
sys_d = c2d(sys_c, dt, 'zoh');
Ad = sys_d.A;
Bd = sys_d.B;

fprintf('Discrete A_d:\n'); disp(Ad);
fprintf('Discrete B_d:\n'); disp(Bd);

%% 5. Process noise setup
% Wind acts as a random force disturbance on velocity.
% Model: w_k ~ N(0, Q_d), only entering the velocity state.
%
% Physically: think of it as a random force F_w with std sigma_F (N),
% producing a velocity kick of (F_w/m)*dt each step.

%sigma_F = 10e4;                      % Std of disturbance force [N] (~20 kN gusts)
%sigma_v = (sigma_F / m) * dt;       % Resulting velocity-kick std [m/s]

sigma_v = 0.5;

Qd = [0,    0;
      0,    sigma_v^2];             % Discrete process noise covariance

%% 6. Closed-loop simulation (discrete-time loop)
T_final = 200;
N       = round(T_final/dt);
t       = (0:N-1)' * dt;

v_ref   = 30;
z_ref   = @(tk) [v_ref*tk; v_ref];

Z  = zeros(N, 2);
Ft = zeros(N, 1);
W  = zeros(N, 2);

z = [0; 0];                         % Initial state

for k = 1:N
    tk = t(k);

    % Control law: F_t = K*(z_ref - z)
    Ft(k) = K * (z_ref(tk) - z);

    % Draw process noise for this step
    % Note: chol(Qd) won't work directly because Qd is singular (rank 1),
    % so we just sample the nonzero component directly.
    w = [0; sigma_v * randn];
    W(k,:) = w.';

    Z(k,:) = z.';

    % Propagate one step: z_{k+1} = Ad*z + Bd*[F_t; 1] + w
    z = Ad*z + Bd*[Ft(k); 1] + w;
end

%% 7. Plot results
figure('Position', [100 100 900 900]);

subplot(2,1,1);
plot(t, Z(:,2), 'b', 'LineWidth', 1.5); hold on;
yline(v_ref, 'r--', 'LineWidth', 1.5);
ylabel('Velocity [m/s]');
title('Discrete-Time Train Controller with Process Noise');
legend('Actual', 'Reference', 'Location', 'southeast');
grid on;

subplot(2,1,2);
plot(t, W(:,2), 'k', 'LineWidth', 1);
ylabel('Noise Kick [m/s]');
xlabel('Time [s]');
grid on;

%% 8. Performance metrics
% After the initial transient, look at the steady-state statistics
steady_idx  = t > 100;
v_mean      = mean(Z(steady_idx, 2));
v_std       = std(Z(steady_idx, 2));
Ft_std      = std(Ft(steady_idx));

fprintf('\n--- Steady-State Performance (t > 100s) ---\n');
fprintf('Mean velocity:        %.3f m/s (ref = %.1f)\n', v_mean, v_ref);
fprintf('Velocity std dev:     %.3f m/s\n', v_std);
fprintf('Throttle std dev:     %.2f kN\n', Ft_std/1000);