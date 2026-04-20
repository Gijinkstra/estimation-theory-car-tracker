%% Train Cruise Controller — Classic Reference Tracking
% State:        z = [x; v]
% Control law:  F_t = K * (z_ref - z)
% Design:       Pole placement for desired wn, zeta

clear; clc; close all;

%% 1. Physical parameters
m     = 4e5;        % Train mass [kg]
b     = 5000;       % Linear drag coefficient [N*s/m]
mu    = 0.0015;     % Rolling friction coefficient [-]
g     = 9.81;       % Gravity [m/s^2]

%% 2. State-space model (Option 2: augmented input)
% dz/dt = A*z + B*u,  where u = [F_t; 1]
A = [0   1;
     0  -b/m];

B = [0       0;
     1/m   -mu*g];

%% 3. Controller design via pole placement
wn   = 0.1;         % Natural frequency [rad/s]
zeta = 1.0;         % Damping ratio (critically damped)

% Desired characteristic polynomial: s^2 + 2*zeta*wn*s + wn^2
desired_poly  = [1, 2*zeta*wn, wn^2];
desired_poles = roots(desired_poly);

fprintf('Desired closed-loop poles:\n');
disp(desired_poles);

% Only the first column of B drives the control input
B_ctrl = B(:,1);

% State feedback gain
K = place(A, B_ctrl, desired_poles);

fprintf('State feedback gain K = [k1, k2]:\n');
disp(K);

%% 4. Closed-loop simulation
v_ref = 30;         % Desired cruise speed [m/s]

tspan = [0 200];
z0    = [0; 0];     % Start at rest

% Reference state: position advances at v_ref, velocity is v_ref
% z_ref(t) = [v_ref*t; v_ref]
z_ref = @(t) [v_ref*t; v_ref];

% Control law: F_t = K * (z_ref - z)
% Full plant input vector: [F_t; 1]
odefun = @(t, z) A*z + B*[ K*(z_ref(t) - z); 1 ];

[t, Z] = ode45(odefun, tspan, z0);

%% 5. Recover control effort over time
Ft = zeros(size(t));
for i = 1:length(t)
    Ft(i) = K * (z_ref(t(i)) - Z(i,:).');
end

%% 6. Plot results
figure('Position', [100 100 900 800]);

subplot(4,1,1);
plot(t, Z(:,2), 'b', 'LineWidth', 2); hold on;
yline(v_ref, 'r--', 'LineWidth', 1.5);
ylabel('Velocity [m/s]');
title('Train Cruise Controller — Classic Tracking Form');
legend('Actual', 'Reference', 'Location', 'southeast');
grid on;

subplot(4,1,2);
plot(t, Z(:,2) - v_ref, 'b', 'LineWidth', 2);
ylabel('Velocity Error [m/s]');
grid on;

subplot(4,1,3);
plot(t, Z(:,1)/1000, 'b', 'LineWidth', 2); hold on;
plot(t, v_ref*t/1000, 'r--', 'LineWidth', 1.5);
ylabel('Position [km]');
legend('Actual', 'Reference', 'Location', 'southeast');
grid on;

subplot(4,1,4);
plot(t, Ft/1000, 'b', 'LineWidth', 2);
ylabel('Throttle Force [kN]');
xlabel('Time [s]');
grid on;

%% 7. Performance metrics
v_final     = Z(end, 2);
ss_error    = v_ref - v_final;
settle_idx  = find(abs(Z(:,2) - v_ref) > 0.02*v_ref, 1, 'last');
if isempty(settle_idx)
    settle_time = 0;
else
    settle_time = t(settle_idx);
end

fprintf('\n--- Performance ---\n');
fprintf('Final velocity:      %.3f m/s\n', v_final);
fprintf('Steady-state error:  %.4f m/s\n', ss_error);
fprintf('2%% settling time:    %.1f s\n', settle_time);
fprintf('Peak throttle force: %.1f kN\n', max(abs(Ft))/1000);