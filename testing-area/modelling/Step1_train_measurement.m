%% Train Cruise Controller — Full State Observer + State Feedback
% Plant:       z_{k+1} = Ad*z + Bd*u + w     (process noise)
%              y_k     = C*z + v             (measurement noise)
% Observer:    zhat_{k+1} = Ad*zhat + Bd*u + L*(y - C*zhat)
% Controller:  F_t = K*(z_ref - zhat)
% Design:      Pole placement for both K and L (acker / place)

clear; clc; close all;
rng(1);             % Reproducible noise

%% 1. Physical parameters
m     = 4e5;        % Train mass [kg]
b     = 5000;       % Linear drag coefficient [N*s/m]
mu    = 0.0015;     % Rolling friction coefficient [-]
g     = 9.81;       % Gravity [m/s^2]

%% 2. Continuous state-space model (Option 2: augmented input)
% State: z = [x; v], Input: u = [F_t; 1]
A = [0   1;
     0  -b/m];

B = [0       0;
     1/m   -mu*g];

C = eye(2);         % Measure both position and velocity
D = zeros(2, 2);

%% 3. Controller design (continuous-time pole placement)
wn_ctrl   = 0.1;    % Controller natural frequency [rad/s]
zeta_ctrl = 1.0;    % Critically damped

ctrl_poly        = [1, 2*zeta_ctrl*wn_ctrl, wn_ctrl^2];
ctrl_poles_cont  = roots(ctrl_poly);

%% 4. Observer design (3x faster than controller)
wn_obs   = 3 * wn_ctrl;
zeta_obs = 1.0;

obs_poly        = [1, 2*zeta_obs*wn_obs, wn_obs^2];
obs_poles_cont  = roots(obs_poly);

%% 5. Discretize
dt    = 0.5;        % Sample time [s]
sys_c = ss(A, B, C, D);
sys_d = c2d(sys_c, dt, 'zoh');
Ad    = sys_d.A;
Bd    = sys_d.B;
Cd    = sys_d.C;

% Map poles from s-plane to z-plane: z = exp(s*dt)
ctrl_poles_disc = exp(ctrl_poles_cont * dt);
obs_poles_disc  = exp(obs_poles_cont  * dt);

%% 6. Compute gains
B_ctrl = Bd(:,1);                               % Only first column is controllable input
K      = place(Ad, B_ctrl, ctrl_poles_disc);    % 1x2 state feedback gain
L      = place(Ad', Cd', obs_poles_disc)';      % 2x2 observer gain

fprintf('Controller gain K:\n'); disp(K);
fprintf('Observer   gain L:\n'); disp(L);

%% 7. Noise setup
% Process noise: random wind force on velocity
sigma_F      = 2e4;                             % Disturbance force std [N]
sigma_v_proc = (sigma_F/m) * dt;                % Velocity kick std [m/s]

% Measurement noise on each sensor
sigma_x_meas = 1.0;                             % Position sensor std [m]
sigma_v_meas = 0.3;                             % Velocity sensor std [m/s]

%% 8. Simulation
T_final = 200;
N       = round(T_final/dt);
t       = (0:N-1)' * dt;

v_ref   = 30;                                   % Cruise speed [m/s]
z_ref   = @(tk) [v_ref*tk; v_ref];

% Preallocate
Z     = zeros(N, 2);    % True state
Zhat  = zeros(N, 2);    % Estimated state
Y     = zeros(N, 2);    % Measurement
Ft    = zeros(N, 1);    % Control

% Initial conditions
z    = [0; 0];          % Plant starts at rest
zhat = [0; 5];          % Deliberately wrong initial estimate, to see convergence

for k = 1:N
    tk = t(k);

    % --- Plant produces a noisy measurement ---
    v_meas = [sigma_x_meas * randn;
              sigma_v_meas * randn];
    y = Cd * z + v_meas;

    % --- Controller acts on the estimate ---
    Ft(k) = K * (z_ref(tk) - zhat);

    % --- Record histories ---
    Z(k,:)    = z.';
    Zhat(k,:) = zhat.';
    Y(k,:)    = y.';

    % --- Observer update ---
    zhat = Ad*zhat + Bd*[Ft(k); 1] + L*(y - Cd*zhat);

    % --- Plant propagates with process noise ---
    w = [0; sigma_v_proc * randn];
    z = Ad*z + Bd*[Ft(k); 1] + w;
end

%% 9. Plots
figure('Position', [100 100 1000 900]);

subplot(1,1,1);
plot(t, Y(:,2), 'Color', [0.6 0.8 1], 'LineWidth', 0.5); hold on;
plot(t, Zhat(:,2), 'g', 'LineWidth', 1.2);
plot(t, Z(:,2), 'b', 'LineWidth', 1.5);
yline(v_ref, 'r--', 'LineWidth', 1.2);
ylabel('Velocity [m/s]');
title('Train with Observer + State Feedback Controller');
legend('Measured', 'Estimated', 'True', 'Reference', 'Location', 'southeast');
grid on;

%% 10. Performance metrics (steady state, t > 100s)
ss = t > 100;
fprintf('\n--- Steady-State Performance (t > 100s) ---\n');
fprintf('Mean velocity:        %.3f m/s (ref = %.1f)\n', mean(Z(ss,2)), v_ref);
fprintf('Velocity std dev:     %.3f m/s\n',  std(Z(ss,2)));
fprintf('Est. error std dev:   %.3f m/s\n',  std(Zhat(ss,2) - Z(ss,2)));
fprintf('Throttle std dev:     %.2f kN\n',   std(Ft(ss))/1000);