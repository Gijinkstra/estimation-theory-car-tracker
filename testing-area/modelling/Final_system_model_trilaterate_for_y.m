%% State-Space Simulation: Vehicle on a Closed Track with Beacon Measurements
%
%  State  x = [s ; v ; l ; vl]
%    s  : arc-length position along the track (closed loop)     [m]
%    v  : longitudinal speed along the track                    [m/s]
%    l  : lateral position, normal to the track centre-line     [m]  (|l|<=l_max)
%    vl : lateral speed                                         [m/s]
%
%  Process model (Euler, step h):
%    s_{k+1}  = ( s_k + h*v_k ) mod L_total                  (closed loop)
%    v_{k+1}  = v_k  + w_v,       w_v ~ N(0, sigma_v^2)       (zero-mean random walk,
%                                                              v_0 comes from x0)
%    l_{k+1}  = l_k  + h*vl_k                                (clamped to +-l_max)
%    vl_{k+1} = vl_k + w_l(l_k), w_l is zero-mean Beta-distributed noise
%                                whose shape (concentration + skew) depends on l.
%
%  Measurement model (3 fixed beacons, Euclidean range - nonlinear in (s,l)):
%    y_i,k = || p(s_k, l_k) - b_i ||_2 + n_i,k ,   n_i,k ~ N(0, sigma_r^2)
%    where p(s,l) : path-coordinates -> world frame (x,y).
%
%  Recovery of s from the 3 beacon ranges:
%    (a) 2-D trilateration  ->  (x_hat, y_hat)
%    (b) section-wise analytical projection onto the 4 track sections
%        (bottom straight, small arc, top straight, big arc). Return the
%        section giving the smallest |l|.
%
%  The track is a closed loop formed by a LARGE semicircle (radius R1) and
%  a SMALL semicircle (radius R2) joined by the two external-tangent
%  straights. With R1 ~= R2 the connecting straights are slightly inclined
%  (angle alpha = asin((R1-R2)/D)), so the big/small arcs subtend
%  (pi + 2*alpha) and (pi - 2*alpha) rad respectively (not exactly pi).

clear; clc; close all; rng(42);

%% ====================== ASSUMPTIONS & CONSTANTS =========================
% Any quantity the user did not specify is assumed here and flagged.

% ---- Simulation ---------------------------------------------------------
h       = 0.05;             % Discretisation step                [s]  
T_sim   = 120;              % Total simulated time               [s]
N       = round(T_sim/h);   % Number of steps

% ---- Track geometry (closed loop) --------------------------------------
R1      = 50;               % Large semicircle radius           [m]  
R2      = 20;               % Small semicircle radius           [m]  
D       = 100;              % Centre-to-centre distance         [m]  
track_w = 4;                % Track width                       [m]  
l_max   = track_w/2;        % Max lateral offset  +-2 m         [m]  

% Derived geometry 
alpha      = asin((R1-R2)/D);           % the angle that the straight line is inclined relative to x axis. 
L_str      = sqrt(D^2 - (R1-R2)^2);     % length of each connecting straight
L_arc2     = (pi - 2*alpha)*R2;         % small-circle arc length (right side)
L_arc1     = (pi + 2*alpha)*R1;         % large-circle arc length (left side)
L_total    = 2*L_str + L_arc1 + L_arc2; % total track length

% The 4 connection (tangent) points between sections. [x y] co-ords
P1_top = [ R1*sin(alpha),      R1*cos(alpha) ];   % big circle, top tangent
P1_bot = [ R1*sin(alpha),     -R1*cos(alpha) ];   % big circle, bottom tangent
P2_top = [ D + R2*sin(alpha),  R2*cos(alpha) ];   % small circle, top tangent
P2_bot = [ D + R2*sin(alpha), -R2*cos(alpha) ];   % small circle, bottom tangent

% Bundle geometry for passing to local functions
geom = struct('R1',R1,'R2',R2,'D',D,'alpha',alpha, ...
              'L_str',L_str,'L_arc1',L_arc1,'L_arc2',L_arc2, ...
              'L_total',L_total, ...
              'P1_bot',P1_bot,'P2_bot',P2_bot,'P1_top',P1_top,'P2_top',P2_top);

% Convention for l: +l is to the LEFT of the direction of motion, which
% in this loop corresponds to the interior of the loop (towards the
% centre-line of the two circles) on every section.

% ---- Process noise ------------------------------------------------------
% Longitudinal noise is zero-mean Gaussian - a simple random walk on v.
% The cruise speed is set by the initial condition v_0 (see x0 below);
% E[v_k] = v_0 for all k, but the variance grows linearly in k so over
% long runs v will wander
sigma_v  = 0.2;            % Std of zero-mean longitudinal speed noise [m/s]

% Lateral (Beta) noise design:
%     X ~ Beta( a(l), b(l) ) ,   w_l = beta_scale * ( X - a/(a+b) )
% So E[w_l] = 0 at every l by construction -- the lateral noise has zero
% mean. Only the SHAPE depends on l. Instead of designing (a, b) from a
% closed-form formula in l, we pick (a, b) DIRECTLY at three anchor
% points in l:
%
%     l = -l_max      (car @ LEFT wall)
%     l =  0          (car @ centre)
%     l = +l_max      (car @ RIGHT wall)
%
% then linearly interpolate a(l) and b(l) between them. The animation
% plot makes this shape-change visible at each time step.
beta_scale = 0.01;          % m/s scale of the lateral noise

% At each of these 3 anchor positions in l, we DIRECTLY specify the Beta
% shape parameters a and b. These are the only free design knobs; every
% other (a, b) is interpolated from these. Edit the numbers below to
% reshape the whole noise design.
%
% Intuition for what (a, b) do:
%   a > b  =>  Beta mass piled toward x = 1  (so w_l = scale*(X-mean) tends positive)
%   a < b  =>  Beta mass piled toward x = 0  (so w_l tends negative)
%   a = b  =>  symmetric Beta about x = 0.5
%   a + b  =>  CONCENTRATION: larger sum means narrower, peakier PDF
%
% Here we choose:
%   at l = -l_max :  a=3, b=1   (skewed toward x=1)
%   at l =  0     :  a=3, b=3   (symmetric)
%   at l = +l_max :  a=1, b=3   (skewed toward x=0)
l_cal = [ -l_max,    0,     +l_max ];      % anchor l values
a_cal = [  3.0,      3.0,    1.0   ];      % Beta 'a' at each anchor
b_cal = [  1.0,      3.0,    3.0   ];      % Beta 'b' at each anchor

% Piecewise-linear interpolation blends between the bracketing anchor pair.
% The anchors span the FULL admissible range [-l_max, +l_max], so the car's
% l (which is clamped to that range) always falls inside the interpolation
% interval and no extrapolation is needed.
a_of_l = @(l) max( interp1(l_cal, a_cal, l, 'linear'));
b_of_l = @(l) max( interp1(l_cal, b_cal, l, 'linear'));

% ---- Measurement (3 beacons) -------------------------------------------
beacons = [ -40,   40;
             60,  -50;
            110,   35 ];                % beacon positions in world frame [m]
Nb      = size(beacons,1);
sigma_r = 0.5;              % range measurement noise std         [m]

% ---- Initial state ------------------------------------------------------
x0 = [ 0;     % s   [m]
       8;    % v   [m/s]   (ASSUMED - approx. 29 km/h)
       0;    % l   [m]
       0 ];  % vl  [m/s]

%% ====================== PRE-ALLOCATION ==================================
X     = zeros(4, N+1);   X(:,1) = x0;
Y     = zeros(Nb, N);          % beacon range measurements
XY    = zeros(2, N+1);         % world-frame positions of vehicle
S_est = zeros(1, N);           % s estimated from beacons each step
L_est = zeros(1, N);

[XY(1,1), XY(2,1)] = sl_to_xy(X(1,1), X(3,1), geom);

%% ====================== SIMULATION LOOP =================================
for k = 1:N
    % Read the current state (s,v,l,vl) at time step k for convenience.
    s  = X(1,k);  v  = X(2,k);
    l  = X(3,k);  vl = X(4,k);

    % --------------------------------------------------------------------
    % (1)  PROCESS-NOISE DRAWS
    % --------------------------------------------------------------------
    % (1a) Longitudinal noise w_v: zero-mean Gaussian.
    % This makes v evolve as a random walk starting from the initial
    % value v_0 given in x0. E[v_k] = v_0 for all k, so the nominal
    % cruise speed is set purely by the initial condition; no drift
    % term is used here.
    w_v = sigma_v * randn;

    % (1b) Lateral noise w_l ~ Beta-based, with shape depending on l.
    % (a, b) are obtained by linear interpolation between the calibration
    % anchors at l = -l_max, 0, +l_max defined above. The noise is then
    % re-centred by subtracting a/(a+b), the true Beta mean, so
    % E[w_l] = 0 at every l.
    a_b  = a_of_l(l);
    b_b  = b_of_l(l);
    Xb   = betarnd(a_b, b_b);
    w_l  = beta_scale * ( Xb - a_b/(a_b + b_b) );   % zero-mean

    % --------------------------------------------------------------------
    % (2)  STATE UPDATE (forward Euler)
    % --------------------------------------------------------------------
    % Arc-length integrates the longitudinal speed; mod L_total wraps the
    % car around the closed loop so it can lap indefinitely.
    s_new  = mod(s  + h*v,  L_total);

    % Longitudinal speed: previous speed + zero-mean Gaussian noise
    % (pure random walk).
    v_new  = v  + w_v;

    % Lateral position: previous + h*lateral_speed.
    l_new  = l  + h*vl;

    % Lateral speed: previous + zero-mean Beta noise (random walk on vl).
    vl_new = vl + w_l;

    % --------------------------------------------------------------------
    % (3)  TRACK-WIDTH CONSTRAINT  (|l| <= l_max)
    % --------------------------------------------------------------------
    % If the Euler step put the car past the inner or outer wall, clamp l
    % to the wall and absorb most of the lateral kinetic energy (coefficient
    % of restitution 0.2 -> mildly bouncy soft wall).
    if l_new >  l_max, l_new =  l_max; vl_new = -0.2*vl_new; end
    if l_new < -l_max, l_new = -l_max; vl_new = -0.2*vl_new; end

    % Store the new state vector into the history array.
    X(:,k+1) = [s_new; v_new; l_new; vl_new];

    % --------------------------------------------------------------------
    % (4)  MAP PATH COORDINATES (s,l) TO THE WORLD FRAME (x,y)
    % --------------------------------------------------------------------
    % sl_to_xy figures out which of the 4 track sections s falls on and
    % applies the corresponding (nonlinear) map. The world-frame position
    % is what the beacons actually see.
    [xw, yw] = sl_to_xy(s_new, l_new, geom);
    XY(:,k+1) = [xw; yw];

    % --------------------------------------------------------------------
    % (5)  SIMULATE THE 3 BEACON RANGE MEASUREMENTS
    % --------------------------------------------------------------------
    % Each beacon reports Euclidean distance from the car to the beacon,
    % corrupted by independent zero-mean Gaussian noise of std sigma_r.
    % This measurement is NONLINEAR in the state (s,l).
    for i = 1:Nb
        Y(i,k) = sqrt( (xw - beacons(i,1))^2 + (yw - beacons(i,2))^2 ) ...
                 + sigma_r*randn;
    end

    % --------------------------------------------------------------------
    % (6)  INVERT THE MEASUREMENT TO RECOVER s
    % --------------------------------------------------------------------
    % Step A: 2-D trilateration from the 3 noisy ranges gives an estimate
    %         (x_hat, y_hat) of the car's world-frame position.
    % Step B: xy_to_sl projects (x_hat, y_hat) onto each of the 4 track
    %         sections analytically and returns the section yielding the
    %         smallest |l|, giving the recovered arc-length s_hat.
    [x_hat, y_hat] = trilaterate(Y(:,k), beacons);
    [s_hat, l_hat] = xy_to_sl(x_hat, y_hat, geom);
    S_est(k) = s_hat;
    L_est(k) = l_hat;
end

%% ====================== POST-PROCESSING & PLOTS =========================
% Draw the track (centre line, inner and outer walls)
s_grid = linspace(0, L_total, 1000);
xy_c   = zeros(2,numel(s_grid));
xy_in  = zeros(2,numel(s_grid));
xy_out = zeros(2,numel(s_grid));
for i = 1:numel(s_grid)
    [xy_c(1,i),   xy_c(2,i)]   = sl_to_xy(s_grid(i),      0, geom);
    [xy_in(1,i),  xy_in(2,i)]  = sl_to_xy(s_grid(i),  l_max, geom);
    [xy_out(1,i), xy_out(2,i)] = sl_to_xy(s_grid(i), -l_max, geom);
end

figure('Position',[80 80 1100 720]);

% --- Track & trajectory ---
subplot(2,2,[1 3]); hold on; axis equal; grid on;
plot(xy_c(1,:),   xy_c(2,:),   'k--', 'LineWidth',0.7);
plot(xy_in(1,:),  xy_in(2,:),  'k-',  'LineWidth',1.2);
plot(xy_out(1,:), xy_out(2,:), 'k-',  'LineWidth',1.2);
plot(XY(1,:), XY(2,:), 'b-', 'LineWidth',0.75);
plot(beacons(:,1), beacons(:,2), 'rs', ...
     'MarkerFaceColor','r','MarkerSize',10);
for i = 1:Nb
    text(beacons(i,1)+2, beacons(i,2)+2, sprintf('B_%d',i), ...
         'Color','r','FontWeight','bold');
end
title('Track and simulated trajectory');
xlabel('x [m]'); ylabel('y [m]');
legend({'centre line','inner wall','outer wall','vehicle path','beacons'}, ...
       'Location','southoutside','Orientation','horizontal');

% --- Longitudinal position (true vs. recovered from beacons) ---
t = (0:N)*h;
subplot(2,2,2); hold on; grid on;
plot(t,             X(1,:), 'b-', 'LineWidth',1.1);
plot(t(1:end-1),    S_est,  'r.', 'MarkerSize',4);
xlabel('time [s]'); ylabel('s [m]');
legend('true s','s recovered from beacons','Location','best');
title('Longitudinal position along track');

% --- Speed and lateral position ---
subplot(2,2,4); hold on; grid on;
yyaxis left;  plot(t, X(2,:), 'b-', 'LineWidth',1.0); ylabel('v [m/s]');
yyaxis right; plot(t, X(3,:), 'r-', 'LineWidth',1.0); ylabel('l [m]');
xlabel('time [s]');
title('Longitudinal speed and lateral offset');

fprintf('Track length L_total = %.2f m\n', L_total);
fprintf('Simulated laps (approx) : %.2f\n', X(1,end)/L_total + ...
        floor( sum(diff(mod(cumsum(X(2,1:end-1))*h, L_total))<0) ));
fprintf('Mean |s_est - s_true|   = %.3f m\n', ...
        mean(abs(wrap_diff(S_est - X(1,2:end), L_total))));

%% ====================== ANIMATION =======================================
% Step through the simulation and update three live subplots:
%   Left   : top-down view of the track with the car marker and its trail.
%   Middle : PDF of the lateral noise w_l at the car's current l, with the
%            mean (drift) drawn as a vertical line to make the restoring
%            behaviour visible.
%   Right  : full l(t) trace with a moving marker showing where the car is
%            right now, and the +-l_max walls drawn as dashed lines.
% Set DO_ANIMATE to false to skip this block.

DO_ANIMATE  = true;          % set false to skip
frame_dt    = 1.0;           % simulated time between animation frames [s]
playback_dt = 0.5;          % wall-clock delay between frames          [s]

if DO_ANIMATE
    anim_stride = max(1, round(frame_dt/h));
    frame_idx   = 1:anim_stride:N;

    figA = figure('Position',[60 60 1700 600], 'Name','Simulation animation');

    % --- left: track + car ---
    axT = subplot(1,3,1); hold on; axis equal; grid on;
    plot(xy_c(1,:),   xy_c(2,:),   'k--','LineWidth',0.5);
    plot(xy_in(1,:),  xy_in(2,:),  'k-', 'LineWidth',1.2);
    plot(xy_out(1,:), xy_out(2,:), 'k-', 'LineWidth',1.2);
    plot(beacons(:,1),beacons(:,2),'rs','MarkerFaceColor','r','MarkerSize',9);
    for i = 1:Nb
        text(beacons(i,1)+2, beacons(i,2)+2, sprintf('B_%d',i), ...
             'Color','r','FontWeight','bold');
    end
    hTrail = plot(NaN, NaN, 'b-', 'LineWidth',0.6);
    hCar   = plot(NaN, NaN, 'o', 'MarkerFaceColor',[0 0.45 0.85], ...
                                  'MarkerEdgeColor','k','MarkerSize',10);
    xlabel('x [m]'); ylabel('y [m]');
    titleT = title(axT,'');

    % --- middle: Beta noise PDF of w_l at current l ---
    % Under w_l = beta_scale*(X - a/(a+b)), the support of w_l depends on l
    % (it slides as the Beta mean changes). We fix the plot x-range to the
    % worst-case possible noise support so the axis stays stationary.
    axP = subplot(1,3,2); hold on; grid on;
    x_grid = linspace(1e-3, 1-1e-3, 400);
    w_lim  = beta_scale;                              % worst-case |w_l|
    hPDF    = plot(NaN, NaN, 'b-', 'LineWidth',1.5);
    hZero   = plot([0 0], [0 1], 'r-',  'LineWidth',2);        % E[w_l] = 0 always
    hMode   = plot([0 0], [0 1], 'k:',  'LineWidth',1.0);      % mode reference
    xlabel('w_l  [m/s]'); ylabel('pdf(w_l)');
    xlim([-w_lim w_lim]);
    titleP = title(axP,'');
    legend(hZero, {'E[w_l|l] = 0'}, 'Location','northeast');

    % --- right: lateral position l vs time ---
    % The full l(t) trace is drawn once as a faint background line; an
    % animated marker (hLnow) shows the car's CURRENT (t, l). The +-l_max
    % walls are dashed so it's obvious when the car is bouncing off them.
    t_full = (0:N)*h;
    axL = subplot(1,3,3); hold on; grid on; box on;
    plot(t_full, X(3,:), 'Color',[0.6 0.6 0.85], 'LineWidth',0.8);
    plot([t_full(1) t_full(end)], [ l_max  l_max], 'k--', 'LineWidth',0.8);
    plot([t_full(1) t_full(end)], [-l_max -l_max], 'k--', 'LineWidth',0.8);
    plot([t_full(1) t_full(end)], [ 0      0    ], 'k:',  'LineWidth',0.6);
    hLnow = plot(NaN, NaN, 'o', 'MarkerFaceColor',[0.85 0.15 0.15], ...
                                'MarkerEdgeColor','k', 'MarkerSize',9);
    xlim([t_full(1) t_full(end)]);
    ylim([-1.15*l_max  1.15*l_max]);
    xlabel('t [s]'); ylabel('l [m]');
    titleL = title(axL,'');

    for kk = frame_idx
        % Update car + trail
        set(hCar,   'XData', XY(1,kk),       'YData', XY(2,kk));
        set(hTrail, 'XData', XY(1, 1:kk),    'YData', XY(2, 1:kk));
        set(titleT, 'String', sprintf('t = %5.2f s   l = %+5.2f m   v = %5.2f m/s', ...
                                      (kk-1)*h, X(3,kk), X(2,kk)));

        % Recompute Beta params at the current l, matching the loop formula
        % (linear interpolation between the calibration anchors).
        l_now  = X(3,kk);
        a_n    = a_of_l(l_now);
        b_n    = b_of_l(l_now);
        mu_X   = a_n/(a_n + b_n);

        % Change of variables X -> w_l = beta_scale*(X - mu_X)
        % so w_l axis = beta_scale*(x_grid - mu_X), pdf_w = pdf_X/beta_scale
        pdf_X  = betapdf(x_grid, a_n, b_n);
        w_axis = beta_scale * (x_grid - mu_X);
        pdf_w  = pdf_X / beta_scale;

        set(hPDF, 'XData', w_axis, 'YData', pdf_w);
        ymax   = max(pdf_w)*1.15 + 1e-3;
        ylim(axP, [0 ymax]);
        set(hZero, 'YData', [0 ymax]);
        % Mark the mode of w_l if the Beta has one strictly inside (0,1)
        if a_n > 1 && b_n > 1
            mode_X = (a_n - 1)/(a_n + b_n - 2);
            mode_w = beta_scale*(mode_X - mu_X);
            set(hMode, 'XData', [mode_w mode_w], 'YData', [0 ymax], ...
                       'Visible', 'on');
        else
            set(hMode, 'Visible', 'off');
        end

        set(titleP, 'String', sprintf( ...
            'Beta(a=%.2f, b=%.2f) @ l=%+4.2f m   E[w_l]=0   Std=%.3f m/s', ...
            a_n, b_n, l_now, ...
            beta_scale*sqrt(a_n*b_n/((a_n+b_n)^2*(a_n+b_n+1))) ));

        % Update the moving (t, l) marker on the right-hand subplot so it
        % rides along the pre-drawn l(t) trace as the animation progresses.
        set(hLnow, 'XData', (kk-1)*h, 'YData', l_now);
        set(titleL, 'String', sprintf('Lateral position l(t)   |   t=%.2f s, l=%+5.2f m', ...
                                      (kk-1)*h, l_now));

        drawnow;
        pause(playback_dt);
    end
end

%% ====================== LOCAL FUNCTIONS =================================

function [x, y, psi] = sl_to_xy(s, l, g)
%SL_TO_XY  Map path coordinates (s,l) to world-frame (x,y).
%   Also returns the heading psi of the track tangent at s.
%   Sections (in order of increasing s):
%       1. bottom straight : P1_bot -> P2_bot
%       2. small-circle arc (right side) : P2_bot -> P2_top
%       3. top straight    : P2_top -> P1_top
%       4. big-circle arc  (left side)  : P1_top -> P1_bot
    s = mod(s, g.L_total);

    if     s <= g.L_str
        % Section 1: bottom straight
        t  = s / g.L_str;
        p  = g.P1_bot + t*(g.P2_bot - g.P1_bot);
        psi = atan2(g.P2_bot(2)-g.P1_bot(2), g.P2_bot(1)-g.P1_bot(1));

    elseif s <= g.L_str + g.L_arc2
        % Section 2: small arc, CCW from phi=-pi/2+alpha to pi/2-alpha
        ds  = s - g.L_str;
        phi = (-pi/2 + g.alpha) + ds/g.R2;
        p   = [g.D + g.R2*cos(phi),  g.R2*sin(phi)];
        psi = phi + pi/2;

    elseif s <= 2*g.L_str + g.L_arc2
        % Section 3: top straight
        ds = s - g.L_str - g.L_arc2;
        t  = ds / g.L_str;
        p  = g.P2_top + t*(g.P1_top - g.P2_top);
        psi = atan2(g.P1_top(2)-g.P2_top(2), g.P1_top(1)-g.P2_top(1));

    else
        % Section 4: big arc, CCW from phi=pi/2-alpha through pi to 3*pi/2+alpha
        ds  = s - 2*g.L_str - g.L_arc2;
        phi = (pi/2 - g.alpha) + ds/g.R1;
        p   = [g.R1*cos(phi),  g.R1*sin(phi)];
        psi = phi + pi/2;
    end

    % +l is to the LEFT of the heading psi
    x = p(1) - l*sin(psi);
    y = p(2) + l*cos(psi);
end


function [s_est, l_est] = xy_to_sl(x, y, g)
%XY_TO_SL  Project a world-frame point onto the track by checking each of
%          the 4 sections analytically, and return (s,l) of the closest
%          point (smallest |l|).
    cand = zeros(0,3);  % each row: [s_along_track, l_signed, |l|]

    % -- Section 1: bottom straight --
    seg  = g.P2_bot - g.P1_bot;  L = norm(seg);
    tang = seg / L;              nrm = [-tang(2), tang(1)];
    r    = [x y] - g.P1_bot;
    sp   = r*tang.';             lp = r*nrm.';
    if sp >= 0 && sp <= L
        cand(end+1,:) = [sp, lp, abs(lp)];
    end

    % -- Section 2: small arc (right side of small circle) --
    phi = atan2(y, x - g.D);
    phi_s = -pi/2 + g.alpha;   phi_e =  pi/2 - g.alpha;
    if phi >= phi_s && phi <= phi_e
        rm = hypot(x - g.D, y);
        sp = g.L_str + (phi - phi_s)*g.R2;
        lp = g.R2 - rm;              % +l = inward (towards centre of small circle)
        cand(end+1,:) = [sp, lp, abs(lp)];
    end

    % -- Section 3: top straight --
    seg  = g.P1_top - g.P2_top;  L = norm(seg);
    tang = seg / L;              nrm = [-tang(2), tang(1)];
    r    = [x y] - g.P2_top;
    sp   = r*tang.';             lp = r*nrm.';
    if sp >= 0 && sp <= L
        cand(end+1,:) = [g.L_str + g.L_arc2 + sp, lp, abs(lp)];
    end

    % -- Section 4: big arc (left side of big circle) --
    phi = atan2(y, x);
    phi_s = pi/2 - g.alpha;
    phi_e = phi_s + (pi + 2*g.alpha);
    if phi < phi_s - 1e-9, phi = phi + 2*pi; end
    if phi >= phi_s && phi <= phi_e
        rm = hypot(x, y);
        sp = 2*g.L_str + g.L_arc2 + (phi - phi_s)*g.R1;
        lp = g.R1 - rm;              % +l = inward (towards centre of big circle)
        cand(end+1,:) = [sp, lp, abs(lp)];
    end

    if isempty(cand)
        % Point lies outside all sections' projection bands (rare with noise) -
        % fallback to a coarse sampled search on the centre line.
        sg   = linspace(0, g.L_total, 400);
        best = inf; s_est = 0;
        for i = 1:numel(sg)
            [xc, yc] = sl_to_xy(sg(i), 0, g);
            d = (xc-x)^2 + (yc-y)^2;
            if d < best, best = d; s_est = sg(i); end
        end
        l_est = NaN;
    else
        [~, idx] = min(cand(:,3));
        s_est = cand(idx,1);
        l_est = cand(idx,2);
    end
end


function [x, y] = trilaterate(r, b)
%TRILATERATE  2-D trilateration from 3 Euclidean range measurements.
%   r : 3x1 ranges,  b : 3x2 beacon positions (rows are beacons).
%   Obtained by subtracting circle #1's equation from circles #2 and #3,
%   which turns two quadratics into one linear 2x2 system.
    A = [ -2*(b(2,:) - b(1,:));
          -2*(b(3,:) - b(1,:)) ];
    d = [ r(2)^2 - r(1)^2 - sum(b(2,:).^2) + sum(b(1,:).^2);
          r(3)^2 - r(1)^2 - sum(b(3,:).^2) + sum(b(1,:).^2) ];
    p = A \ d;
    x = p(1);  y = p(2);
end


function d = wrap_diff(e, L)
%WRAP_DIFF  Wrap an error signal into (-L/2, L/2] (for a periodic variable of period L).
    d = mod(e + L/2, L) - L/2;
end