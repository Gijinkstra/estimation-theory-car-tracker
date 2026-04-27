%% Bayesian Moving Horizon Estimator (MHE) for Vehicle on a Closed Track
%
%  Sliding-window MAP estimator derived from the FIE formulation in
%  `2_bayesian_fie__1_.ipynb`.  At every discrete time step we solve a
%  small fmincon problem over the most recent (N_mhe + 1) states, using
%  only the last N_mhe beacon-range measurements, and summarise all older
%  information through an arrival cost on the oldest state in the window.
%
%  STATE        x = [s ; v ; l ; vl]
%
%  MHE PROBLEM AT TIME k    (window covers states X_{k-N_mhe}, ..., X_{k})
%    min over X_win
%         (X_win(:,1) - x_arr)' * P_arr_inv * (X_win(:,1) - x_arr)    % arrival
%       + sum_{j=1..N_mhe} ( (v_{j+1}-v_j)^2 / sigma_v^2               % process
%                          + (vl_{j+1}-vl_j)^2 / sigma_l_est^2 )
%       + sum_{j=1..N_mhe} ( y_j - h(X_win(:,j)) )' R_inv ( y_j - h(.) )% meas
%    subject to  s_{j+1} = s_j + h*v_j        (linear equality)
%                l_{j+1} = l_j + h*vl_j       (linear equality)
%                -l_max <= l_j <= +l_max      (bound)
%
%  ARRIVAL COST UPDATE
%    While the window is still filling (k < N_mhe):  x_arr = x0_prior,
%    P_arr = P0 (the original prior) -- this is equivalent to a growing-
%    horizon FIE.
%    Once the window is full, at each step we shift forward by one and set
%        x_arr       <-- X_win_opt(:, 2)       (next window's first state)
%        P_arr_inv   <-- P_arr_ss_inv          (a tuned steady-state cov)
%    This is a simple filter-based arrival cost.  A more principled
%    choice is to update P_arr with the Hessian / a parallel EKF, at the
%    cost of extra implementation work.
%
%  WHY MHE instead of FIE
%    * Per-step cost is O(N_mhe^p) with p ~ 1.5, NOT O(T^p).  So T_sim
%      can be as long as we like without the solver slowing down.
%    * Naturally online/causal: at time k only measurements up to k are
%      used, matching a real-world estimator.

clear; clc; close all; rng(42);

%% ====================== SIMULATION & ESTIMATION SETUP ===================
h       = 0.01;             % step                                 [s]
T_sim   = 50;                % total simulated time                 [s]
N       = round(T_sim/h);   % number of steps
N_mhe   = 20;               % MHE window length
                            % => window uses N_mhe+1 states and N_mhe meas

% ---- Track geometry (identical to reference simulation) ----------------
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
sigma_v    = 0.05;
beta_scale = 0.01;
l_cal = [-l_max, 0, +l_max];
a_cal = [5.0,    50.0,  1.0];
b_cal = [1.0,    50.0,  5.0];
a_of_l = @(l) interp1(l_cal, a_cal, l, 'linear');
b_of_l = @(l) interp1(l_cal, b_cal, l, 'linear');

% ---- Process noise: ESTIMATOR side (Gaussian approximation) ------------
sigma_l_est = 0.01;

% ---- Initial true state ------------------------------------------------
x0_true = [0; 10; 0; 0];     % [s; v; l; vl]

% ---- Beacons (placed around the track for good triangulation) ----------
beacons = [ -30,  60;       % B1
            130,  60;       % B2
             50, -60 ];     % B3
n_beacons = size(beacons,1);
sigma_y   = 1.5;

%% ====================== TRUTH SIMULATION + MEASUREMENTS =================
X_true = zeros(4, N+1);   X_true(:,1) = x0_true;
Y_meas = zeros(n_beacons, N);

for k = 1:N
    s = X_true(1,k); v = X_true(2,k); l = X_true(3,k); vl = X_true(4,k);

    w_v = sigma_v * randn;
    a_b = a_of_l(l); b_b = b_of_l(l);
    Xb  = betarnd(a_b, b_b);
    w_l = beta_scale*(Xb - a_b/(a_b+b_b));

    s_new  = mod(s + h*v, L_total);
    v_new  = v + w_v;
    l_new  = l + h*vl;
    vl_new = vl + w_l;
    if l_new >  l_max, l_new =  l_max; vl_new = -0.2*vl_new; end
    if l_new < -l_max, l_new = -l_max; vl_new = -0.2*vl_new; end
    X_true(:,k+1) = [s_new; v_new; l_new; vl_new];

    [xw, yw] = sl_to_xy(s, l, geom);
    for i = 1:n_beacons
        d = norm([xw, yw] - beacons(i,:));
        Y_meas(i,k) = d + sigma_y*randn;
    end
end

%% ====================== MHE SETUP =======================================
nx = 4;

% ---- Prior on x_0 ------------------------------------------------------
x0_prior = [0; 0; 0; 0];
P0       = diag([10^2, 5^2, 1^2, 0.5^2]);
P0_inv   = inv(P0);

% ---- Steady-state arrival-cost covariance ------------------------------
%   Tighter than P0 because, by the time the window has filled, beacon
%   measurements have already reduced the state uncertainty.  These values
%   are pragmatic; tuning them up/down trades bias against variance.
P_arr_ss     = 0.1*P0;
P_arr_ss_inv = inv(P_arr_ss);

% ---- Process & measurement covariances ---------------------------------
Q     = diag([sigma_v^2, sigma_l_est^2]);
Q_inv = inv(Q);
R     = sigma_y^2 * eye(n_beacons);
R_inv = inv(R);

% ---- Solver options (quiet, fast -- warm-starts help a LOT) ------------
opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 100, ...
    'OptimalityTolerance', 1e-6, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8, ...
    'HessianApproximation','lbfgs');

% ---- Estimate storage --------------------------------------------------
X_est = zeros(nx, N+1);
X_est(:,1) = x0_prior;

% ---- Arrival-cost state (updated as the window shifts) -----------------
x_arr     = x0_prior;
P_arr_inv = P0_inv;

% Previous window's optimum (for warm-starting the next solve)
X_prev = [];

%% ====================== MHE ROLLING LOOP ================================
fprintf('\n=== Running MHE  (N = %d steps,  N_mhe = %d) ===\n', N, N_mhe);
t_loop = tic;

for k = 1:N
    % -- Window size this iteration (grows from 1 measurement up to N_mhe)
    j_start  = max(1, k - N_mhe + 1);           % global idx of 1st meas in win
    n_meas   = k - j_start + 1;                 % # measurements in window
    n_states = n_meas + 1;                      % # states in window
    n_dec    = nx * n_states;

    Y_win = Y_meas(:, j_start:k);               % 3 x n_meas

    % -- Linear equality constraints (s and l dynamics, one per step) ----
    n_eq = 2 * n_meas;
    rows = zeros(6*n_meas, 1);
    cols = zeros(6*n_meas, 1);
    vals = zeros(6*n_meas, 1);
    ip = 0;
    for j = 1:n_meas
        is   = (j-1)*nx + 1;  iv  = (j-1)*nx + 2;
        il   = (j-1)*nx + 3;  ivl = (j-1)*nx + 4;
        isp1 = j*nx     + 1;  ilp1 = j*nx     + 3;

        rows(ip+1:ip+3) = j;
        cols(ip+1:ip+3) = [isp1; is; iv];
        vals(ip+1:ip+3) = [1; -1; -h];
        ip = ip + 3;

        rows(ip+1:ip+3) = n_meas + j;
        cols(ip+1:ip+3) = [ilp1; il; ivl];
        vals(ip+1:ip+3) = [1; -1; -h];
        ip = ip + 3;
    end
    Aeq = sparse(rows, cols, vals, n_eq, n_dec);
    beq = zeros(n_eq, 1);

    % -- Bound constraints: |l_j| <= l_max -------------------------------
    lb = -inf(n_dec, 1);
    ub =  inf(n_dec, 1);
    for j = 1:n_states
        il = (j-1)*nx + 3;
        lb(il) = -l_max;
        ub(il) =  l_max;
    end

    % -- Cost handle ------------------------------------------------------
    costfun = @(z) map_cost(z, n_meas, nx, x_arr, P_arr_inv, ...
                             Q_inv, R_inv, Y_win, beacons, geom);

    % -- Warm-start from previous window (cheapest good guess) -----------
    if isempty(X_prev)
        % First iteration: propagate prior forward with zero noise
        X_guess = zeros(nx, n_states);
        X_guess(:,1) = x_arr;
        for j = 1:(n_states-1)
            X_guess(1,j+1) = X_guess(1,j) + h*X_guess(2,j);
            X_guess(2,j+1) = X_guess(2,j);
            X_guess(3,j+1) = X_guess(3,j) + h*X_guess(4,j);
            X_guess(4,j+1) = X_guess(4,j);
        end
    else
        % Append one forward-propagated state; drop first if window full
        x_next    = X_prev(:, end);
        x_next(1) = x_next(1) + h*x_next(2);
        x_next(3) = x_next(3) + h*x_next(4);
        if size(X_prev, 2) < n_states
            X_guess = [X_prev, x_next];             % still growing
        else
            X_guess = [X_prev(:, 2:end), x_next];   % shift forward
        end
    end
    X_guess(3,:) = max(min(X_guess(3,:), l_max), -l_max);
    z0 = X_guess(:);

    % -- Solve ------------------------------------------------------------
    z_opt = fmincon(costfun, z0, [], [], Aeq, beq, lb, ub, [], opts);
    X_win = reshape(z_opt, nx, n_states);

    % -- Filtering estimate: most recent state in window ------------------
    X_est(:, k+1) = X_win(:, end);

    % -- Shift arrival cost if window is full ----------------------------
    if n_states == N_mhe + 1
        % Next iteration's first state = current window's 2nd state
        x_arr     = X_win(:, 2);
        P_arr_inv = P_arr_ss_inv;
    end

    X_prev = X_win;

    if mod(k, 50) == 0 || k == N
        fprintf('  step %4d / %d   (t = %5.2f s,  elapsed = %.1f s)\n', ...
                k, N, k*h, toc(t_loop));
    end
end
elapsed = toc(t_loop);
fprintf('MHE finished in %.2f s  (%.1f ms/step average)\n', ...
        elapsed, 1000*elapsed/N);

%% ====================== ERROR ANALYSIS ==================================
t = (0:N)*h;
s_est_wrapped = mod(X_est(1, :), L_total);
err_s  = mod(X_true(1, :) - s_est_wrapped + L_total/2, L_total) - L_total/2;
err_v  = X_true(2, :) - X_est(2, :);
err_l  = X_true(3, :) - X_est(3, :);
err_vl = X_true(4, :) - X_est(4, :);

%% ====================== TRACK + WORLD-FRAME TRAJECTORIES ================
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

%% ====================== ANIMATED COMBINED FIGURE ========================
% Layout: 3 rows x 4 cols, with the track plot spanning the entire top row.
%   Row 1 : track + trajectories + beacons + animated current-position dots
%   Row 2 : 4 state plots   (s, v, l, vl)         actual vs estimate
%   Row 3 : 4 error plots   (err_s, err_v, err_l, err_vl)

fig = figure('Position',[60 60 1500 900], 'Name','MHE: Animated estimation', ...
             'Color','w');
tl  = tiledlayout(fig, 3, 4, 'TileSpacing','compact', 'Padding','compact');
title(tl, sprintf('Bayesian MHE   (N_{mhe} = %d,   h = %.3f s,   T_{sim} = %.1f s)', ...
                  N_mhe, h, T_sim), 'FontWeight','bold','FontSize',12);

% ---- Top row: track plot (spans all 4 columns) -------------------------
ax_tr = nexttile(tl, 1, [1, 4]); hold(ax_tr,'on'); axis(ax_tr,'equal'); grid(ax_tr,'on');
plot(ax_tr, xy_c(1,:),   xy_c(2,:),   'k--','LineWidth',0.6);
plot(ax_tr, xy_in(1,:),  xy_in(2,:),  'k-', 'LineWidth',1.2);
plot(ax_tr, xy_out(1,:), xy_out(2,:), 'k-', 'LineWidth',1.2);
plot(ax_tr, beacons(:,1), beacons(:,2), 'ks', 'MarkerSize',12, ...
     'MarkerFaceColor','y', 'LineWidth',1.5);
text(ax_tr, beacons(:,1)+3, beacons(:,2)+3, {'B_1','B_2','B_3'}, ...
     'FontSize',12,'FontWeight','bold');
hT_traj = plot(ax_tr, NaN, NaN, 'b-',  'LineWidth',1.7);
hE_traj = plot(ax_tr, NaN, NaN, 'r--', 'LineWidth',1.2);
hT_dot  = plot(ax_tr, NaN, NaN, 'bo', 'MarkerSize',10, 'MarkerFaceColor','b');
hE_dot  = plot(ax_tr, NaN, NaN, 'rs', 'MarkerSize',10, 'MarkerFaceColor','r');
xlabel(ax_tr, 'x [m]'); ylabel(ax_tr, 'y [m]');
title(ax_tr, 'Track:  actual = blue circle,  estimate = red square');
legend(ax_tr, [hT_traj, hE_traj], {'actual trajectory','MHE estimate'}, ...
       'Location','northeast');

% ---- Middle row: 4 state plots -----------------------------------------
state_T  = {X_true(1,:), X_true(2,:), X_true(3,:), X_true(4,:)};
state_E  = {s_est_wrapped, X_est(2,:), X_est(3,:), X_est(4,:)};
state_yl = {'s [m]','v [m/s]','l [m]','v_l [m/s]'};
state_tt = {'Arc-length s','Long. speed v','Lateral pos. l','Lateral speed v_l'};

ax_s  = gobjects(1,4);
hT_s  = gobjects(1,4);
hE_s  = gobjects(1,4);
for i = 1:4
    ax_s(i) = nexttile(tl, 4 + i); hold(ax_s(i),'on'); grid(ax_s(i),'on');
    hT_s(i) = plot(ax_s(i), NaN, NaN, 'b-',  'LineWidth',1.4);
    hE_s(i) = plot(ax_s(i), NaN, NaN, 'r--', 'LineWidth',1.0);
    if i == 3   % lateral position bounds
        yline(ax_s(i),  l_max, 'k:');
        yline(ax_s(i), -l_max, 'k:');
    end
    xlabel(ax_s(i), 'time [s]'); ylabel(ax_s(i), state_yl{i});
    title(ax_s(i), state_tt{i});
    xlim(ax_s(i), [0, t(end)]);
    % Pre-set y-limits from full data so the axes don't jump during animation
    yall = [state_T{i}(:); state_E{i}(:)];
    ymin = min(yall); ymax = max(yall);
    pad  = 0.05 * max(ymax - ymin, eps);
    ylim(ax_s(i), [ymin - pad, ymax + pad]);
    if i == 1   % small actual/estimate legend on the first state plot
        legend(ax_s(i), [hT_s(i), hE_s(i)], {'actual','estimate'}, ...
               'Location','best','FontSize',8);
    end
end

% ---- Bottom row: 4 error plots -----------------------------------------
err_data = {err_s, err_v, err_l, err_vl};
err_yl   = {'s_{true} - s_{est} [m]', 'v_{true} - v_{est} [m/s]', ...
            'l_{true} - l_{est} [m]', 'v_{l,true} - v_{l,est} [m/s]'};
err_tt   = {'s error','v error','l error','v_l error'};
err_fmt  = {'%.3f m','%.3f m/s','%.3f m','%.4f m/s'};

ax_e   = gobjects(1,4);
hE_err = gobjects(1,4);
for i = 1:4
    ax_e(i) = nexttile(tl, 8 + i); hold(ax_e(i),'on'); grid(ax_e(i),'on');
    yline(ax_e(i), 0, 'r:');
    xline(ax_e(i), N_mhe*h, 'g:', 'window full', ...
          'LabelVerticalAlignment','bottom','FontSize',8);
    hE_err(i) = plot(ax_e(i), NaN, NaN, 'k-', 'LineWidth',1.2);
    xlabel(ax_e(i), 'time [s]'); ylabel(ax_e(i), err_yl{i});
    rmse_i = sqrt(mean(err_data{i}.^2));
    title(ax_e(i), sprintf('%s  (RMSE = %s)', err_tt{i}, ...
                           sprintf(err_fmt{i}, rmse_i)));
    xlim(ax_e(i), [0, t(end)]);
    yl = max(abs(err_data{i})) * 1.1;
    if yl > 0
        ylim(ax_e(i), [-yl, yl]);
    end
end

% ---- Animation loop ----------------------------------------------------
% Cap to ~250 frames so playback is smooth regardless of N.
n_frames = N + 1;
stride   = max(1, round(n_frames / 250));

% ---- Set up video writer (saves animation as MP4) ----------------------
video_stamp    = datestr(now, 'yyyy-mm-dd_HHMMSS');                     %#ok<TNOW1,DATST>
video_filename = fullfile(pwd, sprintf('estimator_animation_%s.mp4', video_stamp));
try
    vw = VideoWriter(video_filename, 'MPEG-4');
catch
    % MPEG-4 not available on this platform -> fall back to AVI
    video_filename = strrep(video_filename, '.mp4', '.avi');
    vw = VideoWriter(video_filename, 'Motion JPEG AVI');
end
vw.FrameRate = 30;     % playback speed of the saved file
vw.Quality   = 90;
open(vw);
fprintf('Recording animation to: %s\n', video_filename);

for k = 1:stride:n_frames
    % Track plot: growing trajectories + current-position markers
    set(hT_traj, 'XData', XY_true(1, 1:k), 'YData', XY_true(2, 1:k));
    set(hE_traj, 'XData', XY_est(1,  1:k), 'YData', XY_est(2,  1:k));
    set(hT_dot,  'XData', XY_true(1, k),   'YData', XY_true(2, k));
    set(hE_dot,  'XData', XY_est(1,  k),   'YData', XY_est(2,  k));

    % State plots: growing actual & estimate lines
    for i = 1:4
        set(hT_s(i), 'XData', t(1:k), 'YData', state_T{i}(1:k));
        set(hE_s(i), 'XData', t(1:k), 'YData', state_E{i}(1:k));
    end

    % Error plots: growing error lines
    for i = 1:4
        set(hE_err(i), 'XData', t(1:k), 'YData', err_data{i}(1:k));
    end

    drawnow;                       % full draw so getframe captures it cleanly
    writeVideo(vw, getframe(fig)); % append this frame to the video
end

% Final frame: ensure everything is fully drawn at t = T_sim
set(hT_traj, 'XData', XY_true(1,:), 'YData', XY_true(2,:));
set(hE_traj, 'XData', XY_est(1,:),  'YData', XY_est(2,:));
set(hT_dot,  'XData', XY_true(1,end), 'YData', XY_true(2,end));
set(hE_dot,  'XData', XY_est(1,end),  'YData', XY_est(2,end));
for i = 1:4
    set(hT_s(i),   'XData', t, 'YData', state_T{i});
    set(hE_s(i),   'XData', t, 'YData', state_E{i});
    set(hE_err(i), 'XData', t, 'YData', err_data{i});
end
drawnow;
writeVideo(vw, getframe(fig));     % capture the final fully-drawn frame
close(vw);
fprintf('Animation saved (%d frames) to: %s\n', ...
        numel(1:stride:n_frames) + 1, video_filename);

%% ------------------ Summary --------------------------------------------
fprintf('\n========= MHE estimation RMSE (all t) =========\n');
fprintf('  s   : %.3f  m\n',   sqrt(mean(err_s.^2)));
fprintf('  v   : %.3f  m/s\n', sqrt(mean(err_v.^2)));
fprintf('  l   : %.3f  m\n',   sqrt(mean(err_l.^2)));
fprintf('  v_l : %.4f m/s\n',  sqrt(mean(err_vl.^2)));

idx_ss = (N_mhe+1):(N+1);
fprintf('\n=== MHE steady-state RMSE  (t > %.2f s, window full) ===\n', N_mhe*h);
fprintf('  s   : %.3f  m\n',   sqrt(mean(err_s(idx_ss).^2)));
fprintf('  v   : %.3f  m/s\n', sqrt(mean(err_v(idx_ss).^2)));
fprintf('  l   : %.3f  m\n',   sqrt(mean(err_l(idx_ss).^2)));
fprintf('  v_l : %.4f m/s\n',  sqrt(mean(err_vl(idx_ss).^2)));
fprintf('================================================\n');

%% ====================== LOCAL FUNCTIONS =================================

function J = map_cost(z, n_meas, nx, x_anchor, P_anchor_inv, Q_inv, R_inv, ...
                       Y_win, beacons, geom)
%MAP_COST  Negative log posterior for a single MHE window.
%   Identical structure to the FIE cost; the only difference is that
%   "x_anchor" / "P_anchor_inv" act as the arrival cost on the first
%   state of the window, and Y_win contains only the measurements inside
%   the window.
%
%   Window convention:
%     * n_states = n_meas + 1 (z contains that many states, flattened)
%     * Measurements Y_win(:, j) correspond to states X(:, j), j=1..n_meas
%     * The LAST state X(:, end) has no measurement (propagated only)

    n_states = n_meas + 1;
    X = reshape(z, nx, n_states);

    % (1) ARRIVAL COST on first state
    dx0 = X(:, 1) - x_anchor;
    J = dx0' * P_anchor_inv * dx0;

    % (2) PROCESS-NOISE PENALTY (s & l are equality-constrained, so only
    %     v and vl carry residuals)
    DV  = X(2, 2:end) - X(2, 1:end-1);
    DVL = X(4, 2:end) - X(4, 1:end-1);
    J = J + Q_inv(1,1)*sum(DV.^2) + Q_inv(2,2)*sum(DVL.^2);

    % (3) MEASUREMENT PENALTY  (vectorised over window and beacons)
    s_arr = X(1, 1:n_meas)';
    l_arr = X(3, 1:n_meas)';
    [xw, yw] = sl_to_xy_vec(s_arr, l_arr, geom);
    bx = beacons(:,1)';
    by = beacons(:,2)';
    dx = xw - bx;
    dy = yw - by;
    d_pred = sqrt(dx.^2 + dy.^2);
    err = Y_win' - d_pred;
    J = J + R_inv(1,1) * sum(err(:).^2);   % R diagonal & isotropic
end

function [x, y, psi] = sl_to_xy(s, l, g)
%SL_TO_XY  (s,l) -> world (x,y); scalar inputs (matches reference).
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
%SL_TO_XY_VEC  Vectorised (s,l) -> (x,y); inputs are column vectors.
    s_vec = mod(s_vec, g.L_total);
    n = numel(s_vec);
    px = zeros(n, 1);  py = zeros(n, 1);  psi = zeros(n, 1);

    m1 = s_vec <= g.L_str;
    if any(m1)
        s1 = s_vec(m1);
        t1 = s1 / g.L_str;
        px(m1) = g.P1_bot(1) + t1*(g.P2_bot(1) - g.P1_bot(1));
        py(m1) = g.P1_bot(2) + t1*(g.P2_bot(2) - g.P1_bot(2));
        psi(m1) = atan2(g.P2_bot(2) - g.P1_bot(2), ...
                        g.P2_bot(1) - g.P1_bot(1));
    end

    m2 = (s_vec > g.L_str) & (s_vec <= g.L_str + g.L_arc2);
    if any(m2)
        ds  = s_vec(m2) - g.L_str;
        phi = (-pi/2 + g.alpha) + ds/g.R2;
        px(m2) = g.D + g.R2*cos(phi);
        py(m2) =       g.R2*sin(phi);
        psi(m2) = phi + pi/2;
    end

    m3 = (s_vec > g.L_str + g.L_arc2) & (s_vec <= 2*g.L_str + g.L_arc2);
    if any(m3)
        ds = s_vec(m3) - g.L_str - g.L_arc2;
        t3 = ds / g.L_str;
        px(m3) = g.P2_top(1) + t3*(g.P1_top(1) - g.P2_top(1));
        py(m3) = g.P2_top(2) + t3*(g.P1_top(2) - g.P2_top(2));
        psi(m3) = atan2(g.P1_top(2) - g.P2_top(2), ...
                        g.P1_top(1) - g.P2_top(1));
    end

    m4 = s_vec > 2*g.L_str + g.L_arc2;
    if any(m4)
        ds  = s_vec(m4) - 2*g.L_str - g.L_arc2;
        phi = (pi/2 - g.alpha) + ds/g.R1;
        px(m4) = g.R1*cos(phi);
        py(m4) = g.R1*sin(phi);
        psi(m4) = phi + pi/2;
    end

    x = px - l_vec.*sin(psi);
    y = py + l_vec.*cos(psi);
end