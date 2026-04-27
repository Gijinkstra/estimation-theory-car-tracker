%% Bayesian Full Information Estimator (FIE) for Vehicle on a Closed Track
%
%  Combines the vehicle dynamics from `Final_system_model_States_only.m`
%  with three beacons that provide noisy Euclidean range measurements,
%  and recovers the state history with a Bayesian Full Information
%  Estimator formulated exactly as in `2_bayesian_fie__1_.ipynb`.
%
%  STATE        x = [s ; v ; l ; vl]
%
%  TRUTH DYNAMICS (used to generate the data; identical to the reference)
%    s_{k+1}  = ( s_k + h*v_k ) mod L_total           (closed loop)
%    v_{k+1}  = v_k  + w_v,           w_v ~ N(0, sigma_v^2)
%    l_{k+1}  = l_k  + h*vl_k,        clamped to +-l_max (wall bounce on vl)
%    vl_{k+1} = vl_k + w_l(l_k),      w_l zero-mean Beta noise (shape varies with l)
%
%  MEASUREMENT MODEL (3 beacons at fixed (x,y) positions b_i)
%    y_{i,k} = || xy(s_k, l_k) - b_i || + nu_{i,k},   nu ~ N(0, sigma_y^2)
%
%  FIE PROBLEM (= MAP over the whole trajectory; matches the notebook)
%    min over X(:,1..T+1)
%        (X(:,1) - x0_prior)' * P0_inv * (X(:,1) - x0_prior)        % prior
%      + sum_{k=1..T} ( (v_{k+1}-v_k)^2 / sigma_v^2                 % process noise
%                     + (vl_{k+1}-vl_k)^2 / sigma_l_est^2 )
%      + sum_{k=1..T} ( y_k - h(x_k) )' * R_inv * ( y_k - h(x_k) )  % measurements
%    subject to (deterministic part of the dynamics, LINEAR equalities)
%        s_{k+1}  = s_k  + h * v_k
%        l_{k+1}  = l_k  + h * vl_k
%    subject to (state constraint, the track-width box)
%        -l_max <= l_k <= +l_max
%
%  NOTES
%    * The lateral process noise w_l is Beta-distributed in the truth;
%      the estimator approximates it as Gaussian with std sigma_l_est.
%    * The wall-bounce model in the truth is NOT replicated in the
%      estimator -- the bound |l|<=l_max is enforced and any required
%      correction is absorbed by the (penalised) lateral process noise.
%    * mod() is dropped from the estimator's s update (s is allowed to
%      drift); h(x) calls sl_to_xy() which wraps internally, so this is
%      consistent.
%    * The beacons, sigma_y, the prior, and T_sim are the main knobs to
%      experiment with.

clear; clc; close all; rng(42);

%% ====================== SIMULATION & ESTIMATION SETUP ===================
h       = 0.05;            % step                                 [s]
T_sim   = 20;               % total simulated time                 [s]
N       = round(T_sim/h);  % number of steps  (T_sim=8s -> N=160)
T       = N;               % FIE horizon (use ALL measurements)

% ---- Track geometry (identical to the reference simulation) ------------
R1      = 50; R2 = 20; D = 100;
track_w = 4;  l_max = track_w/2;
alpha      = asin((R1-R2)/D);
L_str      = sqrt(D^2 - (R1-R2)^2);
L_arc2     = (pi - 2*alpha)*R2;
L_arc1     = (pi + 2*alpha)*R1;
L_total    = 2*L_str + L_arc1 + L_arc2;
P1_top = [ R1*sin(alpha),  R1*cos(alpha)];
P1_bot = [ R1*sin(alpha), -R1*cos(alpha)];
P2_top = [ D + R2*sin(alpha),  R2*cos(alpha)];
P2_bot = [ D + R2*sin(alpha), -R2*cos(alpha)];
geom = struct('R1',R1,'R2',R2,'D',D,'alpha',alpha, ...
              'L_str',L_str,'L_arc1',L_arc1,'L_arc2',L_arc2, ...
              'L_total',L_total, ...
              'P1_bot',P1_bot,'P2_bot',P2_bot,'P1_top',P1_top,'P2_top',P2_top);

% ---- Process noise: TRUTH side (same as reference script) --------------
sigma_v    = 0.2;     % longitudinal-speed Gaussian noise std    [m/s]
beta_scale = 0.01;    % lateral noise scale                      [m/s]
l_cal = [-l_max, 0, +l_max];
a_cal = [5.0,    50.0,  1.0];
b_cal = [1.0,    50.0,  5.0];
a_of_l = @(l) interp1(l_cal, a_cal, l, 'linear');
b_of_l = @(l) interp1(l_cal, b_cal, l, 'linear');

% ---- Process noise: ESTIMATOR side (Gaussian approximation) ------------
% At l=0,  std(w_l) ~= beta_scale*sqrt(ab/((a+b)^2*(a+b+1))) = 0.01*0.0498 ~= 0.0005
% Near walls, std(w_l) grows to ~0.0014. Use a moderate fixed value.
sigma_l_est = 0.0015;  % representative std of w_l                [m/s]

% ---- Initial true state ------------------------------------------------
x0_true = [0; 8; 0; 0];     % [s; v; l; vl]   (~29 km/h, on centre line)

% ---- Beacons (placed around the track for good triangulation) ----------
beacons = [ -30,  60;     % B1: top-left  (outside the big arc)
            130,  60;     % B2: top-right (outside the small arc)
             50, -60 ];   % B3: bottom-centre
n_beacons = size(beacons,1);

% ---- Measurement noise -------------------------------------------------
sigma_y = 0.5;            % range-measurement noise std            [m]

%% ====================== TRUTH SIMULATION + MEASUREMENTS =================
X_true = zeros(4, N+1);   X_true(:,1) = x0_true;
Y_meas = zeros(n_beacons, N);     % measurement at step k uses x_k

for k = 1:N
    s = X_true(1,k); v = X_true(2,k); l = X_true(3,k); vl = X_true(4,k);

    % Process noise (truth)
    w_v = sigma_v * randn;
    a_b = a_of_l(l); b_b = b_of_l(l);
    Xb  = betarnd(a_b, b_b);
    w_l = beta_scale*(Xb - a_b/(a_b+b_b));

    % Forward dynamics (with wall clamping & mod)
    s_new  = mod(s + h*v, L_total);
    v_new  = v + w_v;
    l_new  = l + h*vl;
    vl_new = vl + w_l;
    if l_new >  l_max, l_new =  l_max; vl_new = -0.2*vl_new; end
    if l_new < -l_max, l_new = -l_max; vl_new = -0.2*vl_new; end
    X_true(:,k+1) = [s_new; v_new; l_new; vl_new];

    % Beacon range measurements (use x_k, not x_{k+1})
    [xw, yw] = sl_to_xy(s, l, geom);
    for i = 1:n_beacons
        d = norm([xw, yw] - beacons(i,:));
        Y_meas(i,k) = d + sigma_y*randn;
    end
end

%% ====================== FIE OPTIMISATION SETUP ==========================
% Decision variable z = vec(X), where X is 4 x (T+1).
% Index map: z((k-1)*4 + j) = X(j,k),  k = 1..T+1,  j = 1..4.

nx      = 4;
n_total = nx*(T+1);

% ---- Linear equality constraints (deterministic part of dynamics) ------
% Row k     : s_{k+1} - s_k - h*v_k  = 0
% Row T+k   : l_{k+1} - l_k - h*vl_k = 0
nnz_per_row = 3;
rows = zeros(2*T*nnz_per_row, 1);
cols = zeros(2*T*nnz_per_row, 1);
vals = zeros(2*T*nnz_per_row, 1);
ip = 0;
for k = 1:T
    is   = (k-1)*nx + 1;  iv  = (k-1)*nx + 2;
    il   = (k-1)*nx + 3;  ivl = (k-1)*nx + 4;
    isp1 = k*nx     + 1;  ilp1 = k*nx     + 3;

    % s row
    rows(ip+1:ip+3) = k;
    cols(ip+1:ip+3) = [isp1; is; iv];
    vals(ip+1:ip+3) = [1; -1; -h];
    ip = ip + 3;

    % l row
    rows(ip+1:ip+3) = T + k;
    cols(ip+1:ip+3) = [ilp1; il; ivl];
    vals(ip+1:ip+3) = [1; -1; -h];
    ip = ip + 3;
end
Aeq = sparse(rows, cols, vals, 2*T, n_total);
beq = zeros(2*T, 1);

% ---- Bound constraints: |l_k| <= l_max ---------------------------------
lb = -inf(n_total, 1);
ub =  inf(n_total, 1);
for k = 1:(T+1)
    il = (k-1)*nx + 3;
    lb(il) = -l_max;
    ub(il) =  l_max;
end

% ---- Prior on x_0 (Gaussian) -------------------------------------------
% Deliberately offset & uncertain so the estimator has work to do.
x0_prior = x0_true + [3; -1; 0.5; 0.05];
P0       = diag([10^2, 2^2, 1^2, 0.2^2]);
P0_inv   = inv(P0);

% ---- Process-noise covariance (estimator side, only on v and vl) -------
Q     = diag([sigma_v^2, sigma_l_est^2]);
Q_inv = inv(Q);

% ---- Measurement-noise covariance --------------------------------------
R     = sigma_y^2 * eye(n_beacons);
R_inv = inv(R);

% ---- Cost handle (MAP / negative log-posterior) ------------------------
costfun = @(z) fie_cost(z, T, nx, x0_prior, P0_inv, Q_inv, R_inv, ...
                         Y_meas, beacons, geom);

% ---- Initial guess: prior propagated forward with zero noise -----------
X_guess        = zeros(nx, T+1);
X_guess(:, 1)  = x0_prior;
for k = 1:T
    X_guess(1, k+1) = X_guess(1, k) + h*X_guess(2, k);
    X_guess(2, k+1) = X_guess(2, k);
    X_guess(3, k+1) = X_guess(3, k) + h*X_guess(4, k);
    X_guess(4, k+1) = X_guess(4, k);
end
X_guess(3, :) = max(min(X_guess(3, :), l_max), -l_max);   % respect bounds
z0 = X_guess(:);

% ---- Solver options ----------------------------------------------------
opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','iter-detailed', ...
    'MaxFunctionEvaluations', 1e6, ...
    'MaxIterations', 500, ...
    'OptimalityTolerance', 1e-6, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8, ...
    'HessianApproximation','lbfgs');

fprintf('\n=== Solving FIE ===\n');
fprintf('  decision variables : %d   (states only; w reconstructed)\n', n_total);
fprintf('  linear equalities  : %d   (deterministic s and l updates)\n', size(Aeq,1));
fprintf('  bound constraints  : %d   (|l_k| <= %.2f m)\n\n', T+1, l_max);

tic;
[z_opt, fval, exitflag, output] = ...
    fmincon(costfun, z0, [], [], Aeq, beq, lb, ub, [], opts);
elapsed = toc;
fprintf('\nfmincon finished in %.2f s, exit flag = %d, final cost = %.4g\n', ...
        elapsed, exitflag, fval);

X_est = reshape(z_opt, nx, T+1);

%% ====================== ERROR ANALYSIS ==================================
t = (0:N)*h;

% Wrap estimated s into [0, L_total) for fair comparison with truth
s_est_wrapped = mod(X_est(1, :), L_total);

% Circular distance for s error (handles wrap-around correctly)
err_s  = mod(X_true(1, :) - s_est_wrapped + L_total/2, L_total) - L_total/2;
err_v  = X_true(2, :) - X_est(2, :);
err_l  = X_true(3, :) - X_est(3, :);
err_vl = X_true(4, :) - X_est(4, :);

%% ====================== PLOTS ===========================================

% --- Figure 1: Estimated vs Actual states -------------------------------
figure('Position',[80 80 1300 800], 'Name','FIE: Estimated vs Actual');

subplot(2,2,1);
plot(t, X_true(1,:), 'b-',  'LineWidth',1.6); hold on; grid on;
plot(t, s_est_wrapped, 'r--','LineWidth',1.2);
xlabel('time [s]'); ylabel('s [m]');
title('Arc-length position s');
legend('Actual','Estimate','Location','best');

subplot(2,2,2);
plot(t, X_true(2,:), 'b-',  'LineWidth',1.6); hold on; grid on;
plot(t, X_est(2,:),   'r--','LineWidth',1.2);
xlabel('time [s]'); ylabel('v [m/s]');
title('Longitudinal speed v');
legend('Actual','Estimate','Location','best');

subplot(2,2,3);
plot(t, X_true(3,:), 'b-',  'LineWidth',1.6); hold on; grid on;
plot(t, X_est(3,:),   'r--','LineWidth',1.2);
yline( l_max,'k:'); yline(-l_max,'k:');
xlabel('time [s]'); ylabel('l [m]');
title('Lateral position l');
legend('Actual','Estimate','Location','best');

subplot(2,2,4);
plot(t, X_true(4,:), 'b-',  'LineWidth',1.6); hold on; grid on;
plot(t, X_est(4,:),   'r--','LineWidth',1.2);
xlabel('time [s]'); ylabel('v_l [m/s]');
title('Lateral speed v_l');
legend('Actual','Estimate','Location','best');

% --- Figure 2: Estimation error over time -------------------------------
figure('Position',[120 80 1300 800], 'Name','FIE: Estimation error');

subplot(2,2,1);
plot(t, err_s, 'k-', 'LineWidth',1.2); grid on; yline(0,'r:');
xlabel('time [s]'); ylabel('s_{true} - s_{est} [m]');
title(sprintf('s error  (RMSE = %.3f m)', sqrt(mean(err_s.^2))));

subplot(2,2,2);
plot(t, err_v, 'k-', 'LineWidth',1.2); grid on; yline(0,'r:');
xlabel('time [s]'); ylabel('v_{true} - v_{est} [m/s]');
title(sprintf('v error  (RMSE = %.3f m/s)', sqrt(mean(err_v.^2))));

subplot(2,2,3);
plot(t, err_l, 'k-', 'LineWidth',1.2); grid on; yline(0,'r:');
xlabel('time [s]'); ylabel('l_{true} - l_{est} [m]');
title(sprintf('l error  (RMSE = %.3f m)', sqrt(mean(err_l.^2))));

subplot(2,2,4);
plot(t, err_vl, 'k-', 'LineWidth',1.2); grid on; yline(0,'r:');
xlabel('time [s]'); ylabel('v_{l,true} - v_{l,est} [m/s]');
title(sprintf('v_l error  (RMSE = %.4f m/s)', sqrt(mean(err_vl.^2))));

% --- Figure 3: Track + trajectories + beacons ---------------------------
s_grid = linspace(0, L_total, 1000);
xy_c   = zeros(2, numel(s_grid));
xy_in  = zeros(2, numel(s_grid));
xy_out = zeros(2, numel(s_grid));
for i = 1:numel(s_grid)
    [xy_c(1,i),   xy_c(2,i)]   = sl_to_xy(s_grid(i),    0, geom);
    [xy_in(1,i),  xy_in(2,i)]  = sl_to_xy(s_grid(i),  l_max, geom);
    [xy_out(1,i), xy_out(2,i)] = sl_to_xy(s_grid(i), -l_max, geom);
end
XY_true = zeros(2, N+1);
XY_est  = zeros(2, N+1);
for k = 1:N+1
    [XY_true(1,k), XY_true(2,k)] = sl_to_xy(X_true(1,k), X_true(3,k), geom);
    [XY_est(1,k),  XY_est(2,k)]  = sl_to_xy(X_est(1,k),  X_est(3,k), geom);
end

figure('Position',[160 80 900 700], 'Name','Track and trajectories');
hold on; axis equal; grid on;
plot(xy_c(1,:),   xy_c(2,:),   'k--','LineWidth',0.6);
plot(xy_in(1,:),  xy_in(2,:),  'k-', 'LineWidth',1.2);
plot(xy_out(1,:), xy_out(2,:), 'k-', 'LineWidth',1.2);
hT = plot(XY_true(1,:), XY_true(2,:), 'b-',  'LineWidth',1.7);
hE = plot(XY_est(1,:),  XY_est(2,:),  'r--', 'LineWidth',1.2);
hB = plot(beacons(:,1), beacons(:,2), 'ks', 'MarkerSize',12, ...
          'MarkerFaceColor','y', 'LineWidth',1.5);
text(beacons(:,1)+3, beacons(:,2)+3, ...
     {'B_1','B_2','B_3'}, 'FontSize',12,'FontWeight','bold');
xlabel('x [m]'); ylabel('y [m]');
title('Track, true trajectory, FIE estimate, and beacon locations');
legend([hT hE hB], {'actual','estimated','beacons'}, ...
       'Location','southoutside','Orientation','horizontal');

% ---- Summary -----------------------------------------------------------
fprintf('\n========= FIE estimation RMSE =========\n');
fprintf('  s   : %.3f  m\n',   sqrt(mean(err_s.^2)));
fprintf('  v   : %.3f  m/s\n', sqrt(mean(err_v.^2)));
fprintf('  l   : %.3f  m\n',   sqrt(mean(err_l.^2)));
fprintf('  v_l : %.4f m/s\n',  sqrt(mean(err_vl.^2)));
fprintf('=======================================\n');

%% ====================== LOCAL FUNCTIONS =================================

function J = fie_cost(z, T, nx, x0_prior, P0_inv, Q_inv, R_inv, ...
                       Y_meas, beacons, geom)
%FIE_COST  Negative log posterior (MAP cost).
%   J(z) = ell_x0(x0) + sum ell_w(w_k) + sum ell_v(y_k - h(x_k))

    X = reshape(z, nx, T+1);

    % (1) PRIOR on x_0
    dx0 = X(:, 1) - x0_prior;
    J = dx0' * P0_inv * dx0;

    % (2) PROCESS-NOISE PENALTY
    %   Since the deterministic parts (s, l) are equality-constrained, the
    %   only randomness comes from v_{k+1} - v_k = w_v,k and
    %   vl_{k+1} - vl_k = w_l,k.  Q is diagonal, so this is just a sum of
    %   weighted squares.
    DV  = X(2, 2:end) - X(2, 1:end-1);
    DVL = X(4, 2:end) - X(4, 1:end-1);
    J = J + Q_inv(1,1)*sum(DV.^2) + Q_inv(2,2)*sum(DVL.^2);

    % (3) MEASUREMENT PENALTY (T measurements; y_k <-> x_k)
    %   Vectorised over all time steps and all 3 beacons.
    s_arr = X(1, 1:T)';     % T x 1
    l_arr = X(3, 1:T)';
    [xw, yw] = sl_to_xy_vec(s_arr, l_arr, geom);   % T x 1 each
    bx = beacons(:,1)';     % 1 x n_beacons
    by = beacons(:,2)';
    dx = xw - bx;           % T x n_beacons (broadcast)
    dy = yw - by;
    d_pred = sqrt(dx.^2 + dy.^2);
    err = Y_meas' - d_pred; % T x n_beacons
    J = J + R_inv(1,1) * sum(err(:).^2);   % R diagonal & isotropic
end

function [x, y, psi] = sl_to_xy(s, l, g)
%SL_TO_XY  (s,l) -> world (x,y), scalar inputs (matches reference).
    s = mod(s, g.L_total);
    if     s <= g.L_str
        t  = s / g.L_str;
        p  = g.P1_bot + t*(g.P2_bot - g.P1_bot);
        psi = atan2(g.P2_bot(2)-g.P1_bot(2), g.P2_bot(1)-g.P1_bot(1));
    elseif s <= g.L_str + g.L_arc2
        ds  = s - g.L_str;
        phi = (-pi/2 + g.alpha) + ds/g.R2;
        p   = [g.D + g.R2*cos(phi),  g.R2*sin(phi)];
        psi = phi + pi/2;
    elseif s <= 2*g.L_str + g.L_arc2
        ds = s - g.L_str - g.L_arc2;
        t  = ds / g.L_str;
        p  = g.P2_top + t*(g.P1_top - g.P2_top);
        psi = atan2(g.P1_top(2)-g.P2_top(2), g.P1_top(1)-g.P2_top(1));
    else
        ds  = s - 2*g.L_str - g.L_arc2;
        phi = (pi/2 - g.alpha) + ds/g.R1;
        p   = [g.R1*cos(phi),  g.R1*sin(phi)];
        psi = phi + pi/2;
    end
    x = p(1) - l*sin(psi);
    y = p(2) + l*cos(psi);
end

function [x, y] = sl_to_xy_vec(s_vec, l_vec, g)
%SL_TO_XY_VEC  Vectorised (s,l) -> (x,y); inputs are column vectors of length T.
%   Used inside the FIE cost so the inner measurement loop is fast.
    s_vec = mod(s_vec, g.L_total);
    n     = numel(s_vec);

    px  = zeros(n, 1);  py  = zeros(n, 1);  psi = zeros(n, 1);

    % Section 1: bottom straight  (0 <= s <= L_str)
    m1 = s_vec <= g.L_str;
    if any(m1)
        s1 = s_vec(m1);
        t1 = s1 / g.L_str;
        px(m1) = g.P1_bot(1) + t1*(g.P2_bot(1) - g.P1_bot(1));
        py(m1) = g.P1_bot(2) + t1*(g.P2_bot(2) - g.P1_bot(2));
        psi(m1) = atan2(g.P2_bot(2) - g.P1_bot(2), ...
                        g.P2_bot(1) - g.P1_bot(1));
    end

    % Section 2: small arc
    m2 = (s_vec > g.L_str) & (s_vec <= g.L_str + g.L_arc2);
    if any(m2)
        ds  = s_vec(m2) - g.L_str;
        phi = (-pi/2 + g.alpha) + ds/g.R2;
        px(m2) = g.D + g.R2*cos(phi);
        py(m2) =       g.R2*sin(phi);
        psi(m2) = phi + pi/2;
    end

    % Section 3: top straight
    m3 = (s_vec > g.L_str + g.L_arc2) & ...
         (s_vec <= 2*g.L_str + g.L_arc2);
    if any(m3)
        ds = s_vec(m3) - g.L_str - g.L_arc2;
        t3 = ds / g.L_str;
        px(m3) = g.P2_top(1) + t3*(g.P1_top(1) - g.P2_top(1));
        py(m3) = g.P2_top(2) + t3*(g.P1_top(2) - g.P2_top(2));
        psi(m3) = atan2(g.P1_top(2) - g.P2_top(2), ...
                        g.P1_top(1) - g.P2_top(1));
    end

    % Section 4: big arc
    m4 = s_vec > 2*g.L_str + g.L_arc2;
    if any(m4)
        ds  = s_vec(m4) - 2*g.L_str - g.L_arc2;
        phi = (pi/2 - g.alpha) + ds/g.R1;
        px(m4) = g.R1*cos(phi);
        py(m4) = g.R1*sin(phi);
        psi(m4) = phi + pi/2;
    end

    % Apply lateral offset (+l is to the LEFT of the heading psi)
    x = px - l_vec.*sin(psi);
    y = py + l_vec.*cos(psi);
end