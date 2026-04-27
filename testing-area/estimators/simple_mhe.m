%% Minimal MHE Example -- 1D vehicle with 2 beacons
%
%  Strip-down of the closed-track example to the absolute minimum that
%  still requires an optimiser-based estimator.
%
%  STATE              x = [p ; v]      position [m], velocity [m/s]
%
%  DYNAMICS           p_{k+1} = p_k + h*v_k       (deterministic)
%                     v_{k+1} = v_k + w_v          w_v ~ N(0, sigma_v^2)
%
%  MEASUREMENT        car at (p, 0), beacons at B_i = (B_ix, B_iy)
%                     y_k(i) = sqrt( (p - B_ix)^2 + B_iy^2 ) + nu     (nonlinear!)
%
%  WHAT THE MHE DOES, IN PLAIN ENGLISH
%  ----------------------------------
%  At every time k it solves:
%      "Given the last N_mhe measurements and a Gaussian prior on the
%       OLDEST state in the window (the 'arrival cost'), what trajectory
%       of states best explains the data?"
%  It outputs the LAST state of that trajectory as the estimate at time k,
%  then slides the window forward by one.

clear; clc; close all; rng(0);

%% --- Setup --------------------------------------------------------------
h     = 0.1;             % time step                                    [s]
T_sim = 10;              % total time                                   [s]
N     = round(T_sim/h);  % number of time steps
N_mhe = 5;               % MHE window length (5 measurements)

B = [ 0, 5;              % beacon 1: (x=0, y=5)
     10, 5];             % beacon 2: (x=10, y=5)
n_b = size(B,1);

sigma_v = 0.5;           % process noise on velocity                    [m/s]
sigma_y = 0.1;           % measurement noise on range                   [m]

x0_true = [0; 1];        % truth: start at p=0, moving at v=1

%% --- Simulate truth + measurements --------------------------------------
X_true = zeros(2, N+1);   X_true(:,1) = x0_true;
Y_meas = zeros(n_b, N);

for k = 1:N
    p = X_true(1,k);  v = X_true(2,k);
    X_true(:,k+1) = [p + h*v ;             % deterministic
                     v + sigma_v*randn];   % noisy
    for i = 1:n_b
        d_true = sqrt((p - B(i,1))^2 + B(i,2)^2);
        Y_meas(i,k) = d_true + sigma_y*randn;
    end
end

%% --- MHE ingredients ----------------------------------------------------
% Prior on x_0 (deliberately wrong so we can see the MHE recover)
x0_prior = [3; 0];                    % we THINK it starts at p=3, v=0
P0       = diag([4, 1]);              % loose prior covariance
P0_inv   = inv(P0);

% Tighter steady-state arrival-cost covariance (used once window is full)
P_arr_ss_inv = inv(diag([0.2, 0.1]));

Q_inv = 1/sigma_v^2;                  % process penalty weight (scalar)
R_inv = (1/sigma_y^2)*eye(n_b);       % measurement penalty weight

opts = optimoptions('fmincon', 'Algorithm','interior-point', 'Display','off');

% Storage
X_est = zeros(2, N+1);   X_est(:,1) = x0_prior;
x_arr     = x0_prior;                 % current arrival anchor
P_arr_inv = P0_inv;                   % current arrival info matrix
X_prev    = [];                       % warm-start cache

%% --- MHE rolling loop ---------------------------------------------------
fprintf('  k | window  |   p_est   v_est   |   p_true   v_true\n');
fprintf(' ---+---------+--------------------+-------------------\n');

for k = 1:N
    % (a) Determine window
    j_start  = max(1, k - N_mhe + 1);    % first measurement in window
    n_meas   = k - j_start + 1;          % # measurements (grows then plateaus)
    n_states = n_meas + 1;               % states in window
    n_dec    = 2 * n_states;             % total decision variables

    Y_win = Y_meas(:, j_start:k);

    % (b) Build equality constraints  p_{j+1} = p_j + h*v_j  for j=1..n_meas
    %     Decision vector layout: z = [p_1;v_1; p_2;v_2; ... ; p_n;v_n]
    Aeq = zeros(n_meas, n_dec);
    for j = 1:n_meas
        ip   = (j-1)*2 + 1;   iv   = (j-1)*2 + 2;     % indices of p_j, v_j
        ipp1 =  j   *2 + 1;                            % index of p_{j+1}
        Aeq(j, ipp1) =  1;
        Aeq(j, ip)   = -1;
        Aeq(j, iv)   = -h;
    end
    beq = zeros(n_meas, 1);

    % (c) Cost handle
    costfun = @(z) mhe_cost(z, n_meas, x_arr, P_arr_inv, Q_inv, R_inv, Y_win, B);

    % (d) Warm start: shift previous solution forward; predict one step ahead
    if isempty(X_prev)
        X_guess      = zeros(2, n_states);
        X_guess(:,1) = x_arr;
        for j = 1:n_states-1
            X_guess(:,j+1) = [X_guess(1,j) + h*X_guess(2,j); X_guess(2,j)];
        end
    else
        x_next = [X_prev(1,end) + h*X_prev(2,end); X_prev(2,end)];
        if size(X_prev,2) < n_states
            X_guess = [X_prev, x_next];                  % window still growing
        else
            X_guess = [X_prev(:,2:end), x_next];         % shift forward
        end
    end
    z0 = X_guess(:);

    % (e) Solve the constrained nonlinear least-squares MAP problem
    z_opt = fmincon(costfun, z0, [], [], Aeq, beq, [], [], [], opts);
    X_win = reshape(z_opt, 2, n_states);

    % (f) The estimate at time k is the LAST state in the optimised window
    X_est(:,k+1) = X_win(:,end);

    % (g) Slide arrival cost forward once window is full
    if n_states == N_mhe + 1
        x_arr     = X_win(:,2);            % previous 2nd state -> new anchor
        P_arr_inv = P_arr_ss_inv;          % use tighter steady-state cov
    end

    X_prev = X_win;                        % cache for next warm start

    if k <= 3 || mod(k,20) == 0 || k == N
        fprintf(' %2d | [%2d..%2d] |  %6.2f  %6.2f  |  %6.2f  %6.2f\n', ...
                k, j_start, k, X_est(1,k+1), X_est(2,k+1), ...
                              X_true(1,k+1), X_true(2,k+1));
    end
end

%% --- Plot ---------------------------------------------------------------
t = (0:N)*h;
figure('Position',[100 100 900 500]);
subplot(2,1,1);
plot(t, X_true(1,:),'b-','LineWidth',1.5); hold on; grid on;
plot(t, X_est(1,:), 'r--','LineWidth',1.2);
xline(N_mhe*h,'k:','window full');
ylabel('p [m]'); legend('true','MHE est','Location','best');
title('Position');

subplot(2,1,2);
plot(t, X_true(2,:),'b-','LineWidth',1.5); hold on; grid on;
plot(t, X_est(2,:), 'r--','LineWidth',1.2);
xline(N_mhe*h,'k:','window full');
xlabel('t [s]'); ylabel('v [m/s]'); legend('true','MHE est','Location','best');
title('Velocity');

%% ========================================================================
%% MAP COST FOR ONE WINDOW
%% ========================================================================
function J = mhe_cost(z, n_meas, x_arr, P_arr_inv, Q_inv, R_inv, Y_win, B)
% Inputs:
%   z          : decision variable, length 2*(n_meas+1)
%   n_meas     : # measurements in this window
%   x_arr      : arrival anchor (best guess for state at start of window)
%   P_arr_inv  : info matrix for arrival cost
%   Q_inv      : 1/sigma_v^2  (penalty on velocity changes)
%   R_inv      : measurement information matrix
%   Y_win      : 2 x n_meas measurements in this window
%   B          : beacon positions
%
% Output:
%   J          : negative log posterior (up to constants) for this window

    n_states = n_meas + 1;
    X = reshape(z, 2, n_states);          % rows: p, v;  cols: time

    % (1) ARRIVAL COST -- prior on first state of the window
    dx0 = X(:,1) - x_arr;
    J = dx0' * P_arr_inv * dx0;

    % (2) PROCESS PENALTY -- smoothness on velocity
    %     (position update is enforced exactly by the equality constraints)
    dv = X(2, 2:end) - X(2, 1:end-1);
    J = J + Q_inv * sum(dv.^2);

    % (3) MEASUREMENT PENALTY -- match predicted ranges to measured ranges
    for j = 1:n_meas
        p_j    = X(1, j);
        d_pred = sqrt((p_j - B(:,1)).^2 + B(:,2).^2);   % 2x1 predicted ranges
        err    = Y_win(:,j) - d_pred;
        J = J + err' * R_inv * err;
    end
end