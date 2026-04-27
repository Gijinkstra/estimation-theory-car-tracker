%% Bayesian Full Information Estimation (FIE / MAP)
%  Car on a 4 m straight road, tracked by beacons along both edges
%  Constant-acceleration dynamics with state-dependent lateral noise

clear; clc; close all;
rng(37);

% Check casadi
if isempty(which('casadi.SX'))
    error(['casadi not found on MATLAB path. ', ...
           'Run: addpath(''/path/to/your/casadi/folder'')']);
end

%% 1. Problem data
nx = 6;      % state: [px, py, vx, vy, ax, ay]
dt = 0.2;    % sampling time (s)
T  = 40;     % horizon length

% Beacons along both road edges
anchors = [  0,  2;   0, -2;
            20,  2;  20, -2;
            40,  2;  40, -2;
            60,  2;  60, -2];
ny = size(anchors, 1);

% Dynamics matrices (constant acceleration, linear)
A = [eye(2),       dt*eye(2),    zeros(2,2);
     zeros(2,2),   eye(2),       dt*eye(2);
     zeros(2,4),                 eye(2)];
G = [zeros(4,2); eye(2)];

% Noise parameters
sigma_long_sq = 0.01;      % longitudinal acceleration noise variance
sigma_lat_sq  = 0.0025;    % lateral acceleration noise variance (smaller)
k_restore     = 0.8;       % lane-keeping gain: bias = -k * p_y

% Measurement noise
R_scalar = 0.04;           % variance per beacon

% Prior on initial state
x0_tilde = zeros(nx, 1);
P0       = diag([1.0, 1.0, 4.0, 1.0, 1.0, 1.0]);
P0_inv   = inv(P0);

% Road half-width
road_half_width = 2.0;

%% 2. Generate data by simulating the true system
x_true = [0; 0; 5; 0; 0; 0];   % origin, cruising at 5 m/s along +x

x_cache = zeros(T+1, nx);
y_cache = zeros(T, ny);
x_cache(1, :) = x_true';

for t = 1:T
    % Simulate measurement
    v_t = sqrt(R_scalar) * randn(ny, 1);
    % Evaluate h at the true state (numeric beacon-distance calc)
    y_t = zeros(ny, 1);
    for i = 1:ny
        y_t(i) = norm(x_true(1:2) - anchors(i, :)') + v_t(i);
    end
    y_cache(t, :) = y_t';

    % Advance true state with state-dependent lateral noise
    p_y_now = x_true(2);
    w_long  = sqrt(sigma_long_sq) * randn;
    w_lat   = -k_restore * p_y_now + sqrt(sigma_lat_sq) * randn;
    w_t     = [w_long; w_lat];
    x_true  = A * x_true + G * w_t;
    x_cache(t+1, :) = x_true';
end

%% 3. Build symbolic FIE problem via casadi
% Decision variable layout:  z = [x_0; x_1; ...; x_T;  w_0; ...; w_{T-1}]
nw      = 2;
n_total = nx * (T + 1) + nw * T;
z_sym   = casadi.SX.sym('z', n_total);

% --- Extract states and noises symbolically ---
x_sym = cell(T+1, 1);
for t = 0:T
    x_sym{t+1} = z_sym(t*nx + 1 : (t+1)*nx);
end
w_sym = cell(T, 1);
w_offset = nx * (T + 1);
for t = 0:T-1
    w_sym{t+1} = z_sym(w_offset + t*nw + 1 : w_offset + (t+1)*nw);
end

% --- Cost function ---
% Prior cost on x0
dx0 = x_sym{1} - x0_tilde;
cost_expr = dx0' * P0_inv * dx0;

% Per-timestep costs
for t = 1:T
    x_t = x_sym{t};
    w_t = w_sym{t};

    % State-dependent lateral noise cost
    p_y    = x_t(2);
    lat_mean = -k_restore * p_y;
    ell_w_t  = w_t(1)^2 / sigma_long_sq + ...
               (w_t(2) - lat_mean)^2 / sigma_lat_sq;

    % Measurement cost (sum of squared residuals / R)
    y_pred = casadi.SX.zeros(ny, 1);
    for i = 1:ny
        y_pred(i) = norm_2(x_t(1:2) - anchors(i, :)');
    end
    residual = y_cache(t, :)' - y_pred;
    ell_v_t  = (residual' * residual) / R_scalar;

    cost_expr = cost_expr + ell_w_t + ell_v_t;
end

% --- Dynamics equality constraints: x_{t+1} - (A*x_t + G*w_t) = 0 ---
dyn_expr = casadi.SX.zeros(nx * T, 1);
for t = 1:T
    dyn_expr((t-1)*nx + 1 : t*nx) = x_sym{t+1} - (A * x_sym{t} + G * w_sym{t});
end

% --- Gradient of cost, Jacobian of constraints ---
grad_cost_expr = gradient(cost_expr, z_sym);
jac_dyn_expr   = jacobian(dyn_expr, z_sym);

% Wrap as casadi Functions
cost_fun      = casadi.Function('cost',     {z_sym}, {cost_expr});
grad_cost_fun = casadi.Function('gradcost', {z_sym}, {grad_cost_expr});
dyn_fun       = casadi.Function('dyn',      {z_sym}, {dyn_expr});
jac_dyn_fun   = casadi.Function('jacdyn',   {z_sym}, {jac_dyn_expr});

%% 4. MATLAB-callable wrappers for fmincon
% fmincon expects:
%   [f, gradf]      = objective(z)           (when specifying 'SpecifyObjectiveGradient')
%   [c, ceq, gradc, gradceq] = nonlcon(z)    (when specifying 'SpecifyConstraintGradient')

function [f, gradf] = fie_objective(z, cost_fun, grad_cost_fun)
    f     = full(cost_fun(z));
    gradf = full(grad_cost_fun(z));
end

function [c, ceq, gradc, gradceq] = fie_constraints(z, dyn_fun, jac_dyn_fun)
    c   = [];        % no inequality constraints (bounds handle the road)
    ceq = full(dyn_fun(z));
    if nargout > 2
        gradc   = [];
        % fmincon expects the Jacobian TRANSPOSED:
        % rows = variables, columns = constraints
        gradceq = full(jac_dyn_fun(z))';
    end
end

obj_handle     = @(z) fie_objective(z, cost_fun, grad_cost_fun);
nonlcon_handle = @(z) fie_constraints(z, dyn_fun, jac_dyn_fun);

%% 5. Bounds: |p_y| <= 2 for every state, everything else unbounded
lb = -inf(n_total, 1);
ub =  inf(n_total, 1);
for t = 0:T
    p_y_idx = t*nx + 2;   % 2nd entry of each state block is p_y
    lb(p_y_idx) = -road_half_width;
    ub(p_y_idx) =  road_half_width;
end

%% 6. Initial guess: simulate with zero noise from prior mean
x_guess = zeros(T+1, nx);
x_guess(1, :) = x0_tilde';
for t = 1:T
    x_guess(t+1, :) = (A * x_guess(t, :)')';
end
w_guess = zeros(T, nw);

z_guess = [reshape(x_guess', [], 1); reshape(w_guess', [], 1)];

%% 7. Solve with fmincon
options = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...
    'SpecifyObjectiveGradient', true, ...
    'SpecifyConstraintGradient', true, ...
    'MaxIterations', 2000, ...
    'MaxFunctionEvaluations', 50000, ...
    'OptimalityTolerance', 1e-8, ...
    'ConstraintTolerance', 1e-8, ...
    'Display', 'iter');

fprintf('Solving FIE optimisation problem...\n');
[z_opt, fval, exitflag, output] = fmincon( ...
    obj_handle, z_guess, ...
    [], [], [], [], ...
    lb, ub, ...
    nonlcon_handle, options);

fprintf('Exit flag: %d\n', exitflag);
fprintf('Final cost: %.3f\n', fval);

%% 8. Unpack results
x_est = reshape(z_opt(1 : nx*(T+1)), nx, T+1)';
w_est = reshape(z_opt(nx*(T+1)+1 : end), nw, T)';

%% 9. Plot trajectory
figure('Position', [100 100 1100 300]);
hold on; grid on;

x_road = [min(anchors(:,1))-5, max(anchors(:,1))+5];
plot(x_road, [ road_half_width,  road_half_width], 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
plot(x_road, [-road_half_width, -road_half_width], 'k-', 'LineWidth', 1, 'HandleVisibility', 'off');
plot(x_road, [0, 0], 'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');

plot(x_est(:,1), x_est(:,2), 'b-', 'LineWidth', 1.4, 'DisplayName', 'FIE estimate');
plot(x_cache(:,1), x_cache(:,2), 'r-', 'LineWidth', 1.4, 'DisplayName', 'True trajectory');

for i = 1:ny
    if i == 1
        plot(anchors(i,1), anchors(i,2), 'kx', 'MarkerSize', 10, ...
            'LineWidth', 2, 'DisplayName', 'Beacons');
    else
        plot(anchors(i,1), anchors(i,2), 'kx', 'MarkerSize', 10, ...
            'LineWidth', 2, 'HandleVisibility', 'off');
    end
end

xlim(x_road);
ylim([-4, 4]);
xlabel('x-coordinate (m)  --  along road');
ylabel('y-coordinate (m)  --  across road');
title('Bayesian FIE: car tracked by beacons with lane-keeping noise model');
legend('Location', 'northeast');

%% 10. Plot estimation errors
figure('Position', [100 450 900 550]);
time = (0:T) * dt;

subplot(3,1,1);
plot(time, x_cache(:,1) - x_est(:,1), 'b', ...
     time, x_cache(:,2) - x_est(:,2), 'r');
legend('x-error', 'y-error'); ylabel('Position error (m)'); grid on;

subplot(3,1,2);
plot(time, x_cache(:,3) - x_est(:,3), 'b', ...
     time, x_cache(:,4) - x_est(:,4), 'r');
legend('vx-error', 'vy-error'); ylabel('Velocity error (m/s)'); grid on;

subplot(3,1,3);
plot(time, x_cache(:,5) - x_est(:,5), 'b', ...
     time, x_cache(:,6) - x_est(:,6), 'r');
legend('ax-error', 'ay-error'); ylabel('Acceleration error (m/s^2)');
xlabel('Time (s)'); grid on;
sgtitle('Estimation error (truth - estimate)');

%% 11. Plot inferred process noise
figure('Position', [100 1050 900 300]);
time_w = (0:T-1) * dt;
plot(time_w, w_est(:,1), 'b', 'DisplayName', 'w_x (longitudinal)');
hold on;
plot(time_w, w_est(:,2), 'r', 'DisplayName', 'w_y (lateral)');
yline(0, 'k-', 'HandleVisibility', 'off', 'LineWidth', 0.5);
xlabel('Time (s)');
ylabel('Inferred noise (m/s^2)');
title('Estimated process noise (driver inputs)');
legend('Location', 'best');
grid on;