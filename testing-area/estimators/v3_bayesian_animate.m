%% Bayesian MHE -- Practical assessment with real-time animation
%
%  Adapted from v2_bayesian_mhe.m for a real-world implementation check
%  with the sensor specifications you provided:
%
%      Beacon update rate ........ 100 Hz   ( h = 0.01 s )
%      Beacon range  std ......... 1.5  m   ( sigma_y, was 0.5 m )
%
%  WHAT THIS SCRIPT DOES
%      1. Simulates the closed-track vehicle truth at the new sample rate.
%      2. Generates 100 Hz beacon-range measurements with sigma_y = 1.5 m.
%      3. Runs the same MHE you wrote, sized for 100 Hz operation.
%      4. Animates a "live cockpit" view: bird's-eye track + four time-
%         series panels showing actual (blue) vs estimate (red) updating in
%         lock-step.  A status box prints sim time, last solve time, and
%         live RMSE.
%      5. Reports per-step solve time and RMSE so you can judge whether
%         100 Hz operation is feasible on your target hardware.
%
%  REAL-TIME FEASIBILITY
%      For 100 Hz operation the MHE must complete each fmincon solve in
%      under 10 ms on the target machine.  The summary at the end prints
%      mean / median / 95th-percentile / max solve times.  If the 95th
%      percentile is comfortably below 10 ms the estimator is implementable
%      at 100 Hz; if not, options are:
%        (a) shrink N_mhe   (less horizon, faster solve, worse smoothing),
%        (b) decimate MHE rate (e.g. solve at 50 Hz using last 2 meas/step),
%        (c) replace fmincon with a custom Gauss-Newton or KKT solver
%            (orders of magnitude faster for this small QP-like problem).
%
%  PROCESS-NOISE SCALING
%      The original script's sigma_v, beta_scale, sigma_l_est are *per-
%      step* values calibrated for h = 0.1 s.  At h = 0.01 s we scale them
%      by sqrt(h / h_ref) so the *physical* (per-second) noise level of
%      the truth is unchanged -- otherwise simply running at 10x rate
%      would make the truth wander 10x more violently per second, which
%      is not what a faster sensor would actually deliver.

clear; clc; close all; rng(42);

%% =================== USER-FACING FLAGS ==================================
ANIMATE          = true;    % set false for a fast batch run (no live plot)
ANIM_DECIMATE    = 5;       % redraw every Nth step (1 = every step, smoother but slower)
ANIM_PAUSE       = 0;       % seconds to pause between redraws (0 = as-fast-as-possible)
SHOW_FINAL_PLOTS = true;    % static post-run plots (states, errors, track)

%% =================== SENSOR & TIMING SPECS  =============================
h        = 0.01;            % beacon period -> MHE step                 [s]
beacon_rate_Hz = 1/h;       % => 100 Hz
T_sim    = 30;              % simulated time (shorter than original     [s]
                            % to keep the animation tractable; bump up
                            % for a longer feasibility / RMSE study)
N        = round(T_sim/h);  % => 3000 steps for T_sim=30s, h=0.01s

% --- Beacon range noise: the headline practical number ------------------
sigma_y  = 1.5;             % [m]  std of each beacon range measurement
                            % (set to sqrt(1.5) if you meant variance=1.5 m^2)

% --- MHE window ---------------------------------------------------------
% At 100 Hz with 3x noisier beacons we need MORE measurements per window
% than the original (which used 20 meas at sigma_y=0.5).  Variance of an
% averaged range estimate scales as sigma_y^2 / N_mhe; matching the
% original's information content (20/0.25 = 80) gives N_mhe ~ 80*1.5^2 = 180.
% In practice 50 is a good compromise -- enough horizon for smoothing,
% small enough that fmincon stays under the 10 ms budget.
N_mhe    = 50;              % 0.5 s window (50 measurements per beacon)

%% =================== TRACK GEOMETRY (unchanged) =========================
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

%% =================== PROCESS NOISE  =====================================
% Per-step values calibrated for h_ref = 0.1 s in the original script.
% Scale by sqrt(h/h_ref) so the per-second physical noise is unchanged.
h_ref       = 0.1;
noise_scale = sqrt(h / h_ref);

sigma_v     = 0.2  * noise_scale;       % truth: per-step longitudinal accel
beta_scale  = 0.01 * noise_scale;       % truth: lateral kick magnitude
sigma_l_est = 0.0015 * noise_scale;     % estimator: Gaussian approx of beta

l_cal = [-l_max, 0, +l_max];
a_cal = [5.0,    50.0,  1.0];
b_cal = [1.0,    50.0,  5.0];
a_of_l = @(l) interp1(l_cal, a_cal, l, 'linear');
b_of_l = @(l) interp1(l_cal, b_cal, l, 'linear');

% Initial true state
x0_true = [0; 8; 0; 0];     % [s; v; l; vl]   (8 m/s ~ 29 km/h)

% Beacons (same triangle as original)
beacons = [ -30,  60;       % B1
            130,  60;       % B2
             50, -60 ];     % B3
n_beacons = size(beacons,1);

%% =================== TRUTH SIMULATION + MEASUREMENTS ====================
fprintf('Simulating truth (%.0f Hz, %.0f s, %d steps) ...\n', ...
        beacon_rate_Hz, T_sim, N);
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
        Y_meas(i,k) = d + sigma_y*randn;     % <-- 1.5 m std noise here
    end
end

%% =================== MHE SETUP ==========================================
nx = 4;

% Prior on x_0 (deliberately offset to test convergence)
x0_prior = x0_true + [3; -1; 0.5; 0.05];
P0       = diag([10^2, 2^2, 1^2, 0.2^2]);
P0_inv   = inv(P0);

% Steady-state arrival-cost covariance
% NOTE: with sigma_y = 1.5 m the steady-state uncertainty is HIGHER than
% in the original.  Loosen P_arr_ss accordingly so the arrival cost does
% not over-trust the previous window's estimate.
P_arr_ss     = diag([1.5^2, 1.0^2, 0.5^2, 0.2^2]);
P_arr_ss_inv = inv(P_arr_ss);

Q     = diag([sigma_v^2, sigma_l_est^2]);
Q_inv = inv(Q);
R     = sigma_y^2 * eye(n_beacons);
R_inv = inv(R);

opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','off', ...
    'MaxFunctionEvaluations', 1e5, ...
    'MaxIterations', 100, ...
    'OptimalityTolerance', 1e-6, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8, ...
    'HessianApproximation','lbfgs');

X_est       = zeros(nx, N+1);
X_est(:,1)  = x0_prior;
solve_ms    = zeros(N, 1);     % per-step solve time

x_arr     = x0_prior;
P_arr_inv = P0_inv;
X_prev    = [];

%% =================== ANIMATION SETUP ====================================
if ANIMATE
    [animH, t] = setup_animation(geom, beacons, N, h, l_max, ...
                                  sigma_y, beacon_rate_Hz, N_mhe);
else
    t = (0:N)*h;
end

%% =================== MHE ROLLING LOOP  (with live animation) ============
fprintf('\n=== Running MHE  (rate = %g Hz, sigma_y = %.2f m, N_mhe = %d) ===\n', ...
        beacon_rate_Hz, sigma_y, N_mhe);
t_loop = tic;

for k = 1:N
    j_start  = max(1, k - N_mhe + 1);
    n_meas   = k - j_start + 1;
    n_states = n_meas + 1;
    n_dec    = nx * n_states;
    Y_win    = Y_meas(:, j_start:k);

    % Linear equality constraints
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

    lb = -inf(n_dec, 1);
    ub =  inf(n_dec, 1);
    for j = 1:n_states
        il = (j-1)*nx + 3;
        lb(il) = -l_max;
        ub(il) =  l_max;
    end

    costfun = @(z) map_cost(z, n_meas, nx, x_arr, P_arr_inv, ...
                             Q_inv, R_inv, Y_win, beacons, geom);

    if isempty(X_prev)
        X_guess = zeros(nx, n_states);
        X_guess(:,1) = x_arr;
        for j = 1:(n_states-1)
            X_guess(1,j+1) = X_guess(1,j) + h*X_guess(2,j);
            X_guess(2,j+1) = X_guess(2,j);
            X_guess(3,j+1) = X_guess(3,j) + h*X_guess(4,j);
            X_guess(4,j+1) = X_guess(4,j);
        end
    else
        x_next    = X_prev(:, end);
        x_next(1) = x_next(1) + h*x_next(2);
        x_next(3) = x_next(3) + h*x_next(4);
        if size(X_prev, 2) < n_states
            X_guess = [X_prev, x_next];
        else
            X_guess = [X_prev(:, 2:end), x_next];
        end
    end
    X_guess(3,:) = max(min(X_guess(3,:), l_max), -l_max);
    z0 = X_guess(:);

    % --- The actual solve (timed) ---------------------------------------
    t_solve = tic;
    z_opt = fmincon(costfun, z0, [], [], Aeq, beq, lb, ub, [], opts);
    solve_ms(k) = toc(t_solve) * 1000;

    X_win = reshape(z_opt, nx, n_states);
    X_est(:, k+1) = X_win(:, end);

    if n_states == N_mhe + 1
        x_arr     = X_win(:, 2);
        P_arr_inv = P_arr_ss_inv;
    end
    X_prev = X_win;

    % --- Live animation update ------------------------------------------
    if ANIMATE && (mod(k, ANIM_DECIMATE) == 0 || k == N)
        update_animation(animH, k, t, X_true, X_est, geom, ...
                         solve_ms(1:k), L_total, h);
        if ANIM_PAUSE > 0
            pause(ANIM_PAUSE);
        else
            drawnow limitrate;
        end
    end

    if mod(k, 500) == 0 || k == N
        fprintf('  step %5d / %d   (sim t = %5.2f s,  wall = %.1f s, last solve = %.2f ms)\n', ...
                k, N, k*h, toc(t_loop), solve_ms(k));
    end
end
elapsed = toc(t_loop);
fprintf('MHE finished in %.2f s wall-clock  (sim time = %.2f s)\n', elapsed, T_sim);
fprintf('Real-time factor: %.2fx (>1 means estimator is faster than real time)\n', ...
        T_sim / elapsed);

%% =================== ERROR ANALYSIS =====================================
s_est_wrapped = mod(X_est(1, :), L_total);
err_s  = mod(X_true(1, :) - s_est_wrapped + L_total/2, L_total) - L_total/2;
err_v  = X_true(2, :) - X_est(2, :);
err_l  = X_true(3, :) - X_est(3, :);
err_vl = X_true(4, :) - X_est(4, :);

idx_ss = (N_mhe+1):(N+1);

%% =================== PRACTICALITY SUMMARY ===============================
fprintf('\n========================================================\n');
fprintf(' PRACTICALITY ASSESSMENT @ %g Hz, sigma_y = %.2f m\n', beacon_rate_Hz, sigma_y);
fprintf('========================================================\n');
fprintf(' Per-step solve time (fmincon):\n');
fprintf('   mean      : %6.2f ms\n', mean(solve_ms));
fprintf('   median    : %6.2f ms\n', median(solve_ms));
fprintf('   95th pct  : %6.2f ms\n', prctile(solve_ms, 95));
fprintf('   99th pct  : %6.2f ms\n', prctile(solve_ms, 99));
fprintf('   max       : %6.2f ms\n', max(solve_ms));
fprintf('   budget    :  10.00 ms  (for 100 Hz operation)\n');

frac_over = mean(solve_ms > 1000*h) * 100;
fprintf('   solves over budget : %.1f %%\n', frac_over);
if prctile(solve_ms, 95) < 1000*h
    fprintf('   VERDICT  : feasible at 100 Hz on this machine\n');
elseif median(solve_ms) < 1000*h
    fprintf('   VERDICT  : marginal -- median fits but tail exceeds budget\n');
else
    fprintf('   VERDICT  : NOT feasible at 100 Hz with current solver/N_mhe\n');
end

fprintf('\n Steady-state RMSE (t > %.2f s, after window has filled):\n', N_mhe*h);
fprintf('   s   : %.3f  m\n',   sqrt(mean(err_s(idx_ss).^2)));
fprintf('   v   : %.3f  m/s\n', sqrt(mean(err_v(idx_ss).^2)));
fprintf('   l   : %.3f  m\n',   sqrt(mean(err_l(idx_ss).^2)));
fprintf('   v_l : %.4f m/s\n',  sqrt(mean(err_vl(idx_ss).^2)));
fprintf('========================================================\n\n');

%% =================== STATIC POST-RUN PLOTS ==============================
if SHOW_FINAL_PLOTS
    make_static_plots(t, X_true, X_est, s_est_wrapped, err_s, err_v, err_l, err_vl, ...
                      N_mhe, h, l_max, geom, beacons, L_total, solve_ms);
end

%% ============================ LOCAL FUNCTIONS ===========================

function J = map_cost(z, n_meas, nx, x_anchor, P_anchor_inv, Q_inv, R_inv, ...
                       Y_win, beacons, geom)
    n_states = n_meas + 1;
    X = reshape(z, nx, n_states);
    dx0 = X(:, 1) - x_anchor;
    J = dx0' * P_anchor_inv * dx0;
    DV  = X(2, 2:end) - X(2, 1:end-1);
    DVL = X(4, 2:end) - X(4, 1:end-1);
    J = J + Q_inv(1,1)*sum(DV.^2) + Q_inv(2,2)*sum(DVL.^2);
    s_arr = X(1, 1:n_meas)';
    l_arr = X(3, 1:n_meas)';
    [xw, yw] = sl_to_xy_vec(s_arr, l_arr, geom);
    bx = beacons(:,1)';
    by = beacons(:,2)';
    dx = xw - bx;
    dy = yw - by;
    d_pred = sqrt(dx.^2 + dy.^2);
    err = Y_win' - d_pred;
    J = J + R_inv(1,1) * sum(err(:).^2);
end

function [H, t] = setup_animation(geom, beacons, N, h, l_max, ...
                                   sigma_y, rate_Hz, N_mhe)
% Build the live "cockpit" figure once.  Returns a struct of handles
% whose XData/YData get updated each step.
    t = (0:N)*h;

    % Track centre-line and edges (precomputed once)
    s_grid = linspace(0, geom.L_total, 1000);
    xy_c   = zeros(2, numel(s_grid));
    xy_in  = zeros(2, numel(s_grid));
    xy_out = zeros(2, numel(s_grid));
    for i = 1:numel(s_grid)
        [xy_c(1,i),   xy_c(2,i)]   = sl_to_xy(s_grid(i),    0, geom);
        [xy_in(1,i),  xy_in(2,i)]  = sl_to_xy(s_grid(i),  l_max, geom);
        [xy_out(1,i), xy_out(2,i)] = sl_to_xy(s_grid(i), -l_max, geom);
    end

    fig = figure('Position',[60 60 1500 850], ...
                 'Name','MHE Real-Time -- 100 Hz, sigma_y = 1.5 m', ...
                 'Color','w');
    tl = tiledlayout(fig, 4, 3, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['MHE live: beacon rate = %g Hz   sigma_y = %.2f m   ' ...
                        'N_{mhe} = %d   (truth blue, estimate red)'], ...
                        rate_Hz, sigma_y, N_mhe), 'FontWeight','bold');

    % --- Track view: spans all 4 rows of column 1 and 2 -----------------
    ax_track = nexttile(tl, 1, [4, 2]);
    hold(ax_track, 'on'); axis(ax_track, 'equal'); grid(ax_track, 'on');
    plot(ax_track, xy_c(1,:),   xy_c(2,:),   'k--', 'LineWidth', 0.6);
    plot(ax_track, xy_in(1,:),  xy_in(2,:),  'k-',  'LineWidth', 1.2);
    plot(ax_track, xy_out(1,:), xy_out(2,:), 'k-',  'LineWidth', 1.2);
    plot(ax_track, beacons(:,1), beacons(:,2), 'ks', ...
         'MarkerSize',12, 'MarkerFaceColor','y', 'LineWidth',1.5);
    text(ax_track, beacons(:,1)+3, beacons(:,2)+3, ...
         {'B_1','B_2','B_3'}, 'FontSize',12, 'FontWeight','bold');

    % --- Beacon range circles (visualise sensor info live) --------------
    theta = linspace(0, 2*pi, 80);
    H.ring = gobjects(size(beacons,1), 1);
    ring_colors = lines(size(beacons,1));
    for i = 1:size(beacons,1)
        H.ring(i) = plot(ax_track, NaN, NaN, ':', ...
                         'Color', ring_colors(i,:), 'LineWidth', 0.8);
    end
    H.theta = theta;
    H.beacons = beacons;

    % Trails (lines that grow with time)
    H.trail_true = plot(ax_track, NaN, NaN, 'b-',  'LineWidth', 1.4);
    H.trail_est  = plot(ax_track, NaN, NaN, 'r--', 'LineWidth', 1.0);

    % Current vehicle markers
    H.dot_true = plot(ax_track, NaN, NaN, 'bo', ...
                      'MarkerSize',10, 'MarkerFaceColor','b');
    H.dot_est  = plot(ax_track, NaN, NaN, 'r^', ...
                      'MarkerSize',10, 'MarkerFaceColor','r');

    % Legend
    legend(ax_track, [H.trail_true, H.trail_est, H.ring(1)], ...
           {'truth', 'estimate', 'beacon range (\sigma_y)'}, ...
           'Location','southoutside', 'Orientation','horizontal');
    xlabel(ax_track, 'x [m]'); ylabel(ax_track, 'y [m]');
    title(ax_track, 'Bird''s-eye view');

    % Status text in upper corner of track (normalized coords)
    H.txt = text(ax_track, 0.02, 0.98, '', ...
                 'Units','normalized', 'VerticalAlignment','top', ...
                 'FontName','FixedWidth', 'FontSize',9, ...
                 'BackgroundColor','w', 'EdgeColor','k', 'Margin',4);

    % --- Time-series panels: column 3 -----------------------------------
    H.ax_s  = nexttile(tl, 3);
    H.ax_v  = nexttile(tl, 6);
    H.ax_l  = nexttile(tl, 9);
    H.ax_vl = nexttile(tl, 12);

    % Pre-create the line handles with NaN data
    H.s_true  = plot(H.ax_s,  NaN,NaN,'b-', 'LineWidth',1.3); hold(H.ax_s,'on');  grid(H.ax_s,'on');
    H.s_est   = plot(H.ax_s,  NaN,NaN,'r--','LineWidth',1.0);
    ylabel(H.ax_s,'s [m]'); title(H.ax_s,'Arc-length s');

    H.v_true  = plot(H.ax_v,  NaN,NaN,'b-', 'LineWidth',1.3); hold(H.ax_v,'on');  grid(H.ax_v,'on');
    H.v_est   = plot(H.ax_v,  NaN,NaN,'r--','LineWidth',1.0);
    ylabel(H.ax_v,'v [m/s]'); title(H.ax_v,'Longitudinal speed v');

    H.l_true  = plot(H.ax_l,  NaN,NaN,'b-', 'LineWidth',1.3); hold(H.ax_l,'on');  grid(H.ax_l,'on');
    H.l_est   = plot(H.ax_l,  NaN,NaN,'r--','LineWidth',1.0);
    % Reference lines for track edges -- updated per frame so they always span the visible window
    H.l_top = plot(H.ax_l, [0 t(end)], [ l_max  l_max], 'k:', 'LineWidth',0.8);
    H.l_bot = plot(H.ax_l, [0 t(end)], [-l_max -l_max], 'k:', 'LineWidth',0.8);
    ylabel(H.ax_l,'l [m]'); title(H.ax_l,'Lateral position l');
    ylim(H.ax_l, [-1.2*l_max, 1.2*l_max]);    % keep edges always visible

    H.vl_true = plot(H.ax_vl, NaN,NaN,'b-', 'LineWidth',1.3); hold(H.ax_vl,'on'); grid(H.ax_vl,'on');
    H.vl_est  = plot(H.ax_vl, NaN,NaN,'r--','LineWidth',1.0);
    ylabel(H.ax_vl,'v_l [m/s]'); title(H.ax_vl,'Lateral speed v_l');
    xlabel(H.ax_vl,'time [s]');

    drawnow;
end

function update_animation(H, k, t, X_true, X_est, geom, solve_ms_so_far, L_total, h)
% Update the live plot handles after MHE step k.

    % Convert (s,l) -> (x,y) for the current step
    [xw_T, yw_T] = sl_to_xy(X_true(1,k+1), X_true(3,k+1), geom);
    [xw_E, yw_E] = sl_to_xy(mod(X_est(1,k+1), L_total), X_est(3,k+1), geom);

    % Trails (full history -> XY)
    K = k + 1;
    XY_true = zeros(2, K);
    XY_est  = zeros(2, K);
    for j = 1:K
        [XY_true(1,j), XY_true(2,j)] = sl_to_xy(X_true(1,j), X_true(3,j), geom);
        [XY_est(1,j),  XY_est(2,j)]  = sl_to_xy(mod(X_est(1,j), L_total), X_est(3,j), geom);
    end
    set(H.trail_true, 'XData', XY_true(1,:), 'YData', XY_true(2,:));
    set(H.trail_est,  'XData', XY_est(1,:),  'YData', XY_est(2,:));
    set(H.dot_true,   'XData', xw_T, 'YData', yw_T);
    set(H.dot_est,    'XData', xw_E, 'YData', yw_E);

    % Beacon range rings: each ring is the locus of points at the
    % currently-measured range from the beacon (gives a real "pinging
    % sensor" feel)
    for i = 1:size(H.beacons,1)
        d = norm([xw_T, yw_T] - H.beacons(i,:));   % use truth's range
        rx = H.beacons(i,1) + d*cos(H.theta);
        ry = H.beacons(i,2) + d*sin(H.theta);
        set(H.ring(i), 'XData', rx, 'YData', ry);
    end

    % Time series
    tt = t(1:K);
    set(H.s_true,  'XData', tt, 'YData', X_true(1,1:K));
    set(H.s_est,   'XData', tt, 'YData', mod(X_est(1,1:K), L_total));
    set(H.v_true,  'XData', tt, 'YData', X_true(2,1:K));
    set(H.v_est,   'XData', tt, 'YData', X_est(2,1:K));
    set(H.l_true,  'XData', tt, 'YData', X_true(3,1:K));
    set(H.l_est,   'XData', tt, 'YData', X_est(3,1:K));
    set(H.vl_true, 'XData', tt, 'YData', X_true(4,1:K));
    set(H.vl_est,  'XData', tt, 'YData', X_est(4,1:K));

    % Rolling x-limits: grow with the data, with a small floor so plots
    % are not degenerate at t=0 and a small pad so the latest sample is
    % not glued to the right edge.
    t_now      = t(K);
    x_lo       = 0;
    x_hi       = max(t_now * 1.02, 1.0);    % 2% pad, minimum 1 s window
    new_xlim   = [x_lo, x_hi];
    set(H.ax_s,  'XLim', new_xlim);
    set(H.ax_v,  'XLim', new_xlim);
    set(H.ax_l,  'XLim', new_xlim);
    set(H.ax_vl, 'XLim', new_xlim);
    % Stretch the dotted track-edge reference lines to the new x range
    set(H.l_top, 'XData', new_xlim);
    set(H.l_bot, 'XData', new_xlim);

    % Live status box
    e_s  = mod(X_true(1,K) - mod(X_est(1,K),L_total) + L_total/2, L_total) - L_total/2;
    e_v  = X_true(2,K) - X_est(2,K);
    e_l  = X_true(3,K) - X_est(3,K);
    e_vl = X_true(4,K) - X_est(4,K);
    last_solve = solve_ms_so_far(end);
    mean_solve = mean(solve_ms_so_far);
    txt = sprintf([...
        'sim time     : %6.2f s\n' ...
        'last solve   : %6.2f ms\n' ...
        'mean solve   : %6.2f ms\n' ...
        '10 ms budget : %s\n' ...
        '----- live error -----\n' ...
        'e_s   = %+6.3f m\n' ...
        'e_v   = %+6.3f m/s\n' ...
        'e_l   = %+6.3f m\n' ...
        'e_vl  = %+6.3f m/s'], ...
        k*h, last_solve, mean_solve, ...
        ternary(last_solve <= 1000*h, 'OK', 'OVER'), ...
        e_s, e_v, e_l, e_vl);
    set(H.txt, 'String', txt);
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function make_static_plots(t, X_true, X_est, s_est_wrapped, err_s, err_v, err_l, err_vl, ...
                            N_mhe, h, l_max, geom, beacons, L_total, solve_ms)
    % Estimated vs actual
    figure('Position',[80 80 1300 800], 'Name','MHE: Estimated vs Actual');
    subplot(2,2,1); plot(t,X_true(1,:),'b-','LineWidth',1.4); hold on; grid on;
    plot(t,s_est_wrapped,'r--','LineWidth',1); xlabel('time [s]'); ylabel('s [m]');
    title('Arc-length s'); legend('Actual','Estimate');
    subplot(2,2,2); plot(t,X_true(2,:),'b-','LineWidth',1.4); hold on; grid on;
    plot(t,X_est(2,:),'r--','LineWidth',1); xlabel('time [s]'); ylabel('v [m/s]');
    title('Longitudinal speed v'); legend('Actual','Estimate');
    subplot(2,2,3); plot(t,X_true(3,:),'b-','LineWidth',1.4); hold on; grid on;
    plot(t,X_est(3,:),'r--','LineWidth',1); yline(l_max,'k:'); yline(-l_max,'k:');
    xlabel('time [s]'); ylabel('l [m]'); title('Lateral position l'); legend('Actual','Estimate');
    subplot(2,2,4); plot(t,X_true(4,:),'b-','LineWidth',1.4); hold on; grid on;
    plot(t,X_est(4,:),'r--','LineWidth',1); xlabel('time [s]'); ylabel('v_l [m/s]');
    title('Lateral speed v_l'); legend('Actual','Estimate');

    % Errors
    figure('Position',[120 80 1300 800], 'Name','MHE: Estimation error');
    subplot(2,2,1); plot(t,err_s,'k-'); grid on; yline(0,'r:');
    xline(N_mhe*h,'g:','window full');
    xlabel('time [s]'); ylabel('s err [m]');
    title(sprintf('s (RMSE = %.3f m)', sqrt(mean(err_s.^2))));
    subplot(2,2,2); plot(t,err_v,'k-'); grid on; yline(0,'r:');
    xline(N_mhe*h,'g:','window full');
    xlabel('time [s]'); ylabel('v err [m/s]');
    title(sprintf('v (RMSE = %.3f m/s)', sqrt(mean(err_v.^2))));
    subplot(2,2,3); plot(t,err_l,'k-'); grid on; yline(0,'r:');
    xline(N_mhe*h,'g:','window full');
    xlabel('time [s]'); ylabel('l err [m]');
    title(sprintf('l (RMSE = %.3f m)', sqrt(mean(err_l.^2))));
    subplot(2,2,4); plot(t,err_vl,'k-'); grid on; yline(0,'r:');
    xline(N_mhe*h,'g:','window full');
    xlabel('time [s]'); ylabel('v_l err [m/s]');
    title(sprintf('v_l (RMSE = %.4f m/s)', sqrt(mean(err_vl.^2))));

    % Solve time histogram (the key real-time question)
    figure('Position',[160 80 900 500], 'Name','MHE: per-step solve time');
    histogram(solve_ms, 60, 'FaceColor',[0.3 0.5 0.8]); hold on; grid on;
    xline(1000*h,'r-','10 ms budget','LineWidth',1.5,'LabelOrientation','horizontal');
    xline(median(solve_ms),'g--','median');
    xline(prctile(solve_ms,95),'m--','95th pct');
    xlabel('solve time [ms]'); ylabel('count');
    title(sprintf(['fmincon per-step solve time   ' ...
                   '(mean %.2f ms, median %.2f ms, 95%%-tile %.2f ms, max %.2f ms)'], ...
                   mean(solve_ms), median(solve_ms), prctile(solve_ms,95), max(solve_ms)));

    % Track + final trajectories + beacons
    s_grid = linspace(0, L_total, 1000);
    xy_c   = zeros(2, numel(s_grid));
    xy_in  = zeros(2, numel(s_grid));
    xy_out = zeros(2, numel(s_grid));
    for i = 1:numel(s_grid)
        [xy_c(1,i),   xy_c(2,i)]   = sl_to_xy(s_grid(i),    0, geom);
        [xy_in(1,i),  xy_in(2,i)]  = sl_to_xy(s_grid(i),  l_max, geom);
        [xy_out(1,i), xy_out(2,i)] = sl_to_xy(s_grid(i), -l_max, geom);
    end
    XY_true = zeros(2, length(t));
    XY_est  = zeros(2, length(t));
    for k = 1:length(t)
        [XY_true(1,k), XY_true(2,k)] = sl_to_xy(X_true(1,k), X_true(3,k), geom);
        [XY_est(1,k),  XY_est(2,k)]  = sl_to_xy(mod(X_est(1,k),L_total), X_est(3,k), geom);
    end
    figure('Position',[200 80 900 700], 'Name','Track and trajectories');
    hold on; axis equal; grid on;
    plot(xy_c(1,:),xy_c(2,:),'k--','LineWidth',0.6);
    plot(xy_in(1,:),xy_in(2,:),'k-','LineWidth',1.2);
    plot(xy_out(1,:),xy_out(2,:),'k-','LineWidth',1.2);
    hT = plot(XY_true(1,:),XY_true(2,:),'b-','LineWidth',1.6);
    hE = plot(XY_est(1,:),XY_est(2,:),'r--','LineWidth',1.0);
    hB = plot(beacons(:,1),beacons(:,2),'ks','MarkerSize',12,...
              'MarkerFaceColor','y','LineWidth',1.5);
    text(beacons(:,1)+3,beacons(:,2)+3,{'B_1','B_2','B_3'},'FontSize',12,'FontWeight','bold');
    xlabel('x [m]'); ylabel('y [m]');
    title('Track + final trajectories');
    legend([hT hE hB],{'actual','estimated','beacons'},...
           'Location','southoutside','Orientation','horizontal');
end

function [x, y, psi] = sl_to_xy(s, l, g)
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
    s_vec = mod(s_vec, g.L_total);
    n = numel(s_vec);
    px = zeros(n, 1);  py = zeros(n, 1);  psi = zeros(n, 1);

    m1 = s_vec <= g.L_str;
    if any(m1)
        s1 = s_vec(m1);
        t1 = s1 / g.L_str;
        px(m1) = g.P1_bot(1) + t1*(g.P2_bot(1) - g.P1_bot(1));
        py(m1) = g.P1_bot(2) + t1*(g.P2_bot(2) - g.P1_bot(2));
        psi(m1) = atan2(g.P2_bot(2) - g.P1_bot(2), g.P2_bot(1) - g.P1_bot(1));
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
        psi(m3) = atan2(g.P1_top(2) - g.P2_top(2), g.P1_top(1) - g.P2_top(1));
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