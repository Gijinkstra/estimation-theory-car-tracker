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
%    v_{k+1}  = v_k  + w_v,       w_v = (v_target - v_k)*h/tau_v + N(0, sigma_v^2)
%                                  ^ non-zero-mean drift: pulls v toward v_target
%                                  so the car cruises at approximately constant speed
%    l_{k+1}  = l_k  + h*vl_k                                (clamped to +-l_max)
%    vl_{k+1} = vl_k + w_l(l_k), w_l is Beta-distributed noise whose shape AND mean
%                                depend on l: the Beta is aggressively skewed so
%                                E[w_l | l] has opposite sign to l, actively pushing
%                                the car back toward the centre line
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
h       = 0.05;             % Discretisation step                [s]  (ASSUMED)
T_sim   = 120;              % Total simulated time               [s]  (ASSUMED)
N       = round(T_sim/h);   % Number of steps

% ---- Track geometry (closed loop) --------------------------------------
R1      = 50;               % Large semicircle radius           [m]  (ASSUMED)
R2      = 20;               % Small semicircle radius           [m]  (ASSUMED)
D       = 100;              % Centre-to-centre distance         [m]  (ASSUMED)
track_w = 4;                % Track width                       [m]  (GIVEN)
l_max   = track_w/2;        % Max lateral offset  +-2 m         [m]  (GIVEN)

% Derived geometry (do not edit by hand)
alpha      = asin((R1-R2)/D);
L_str      = sqrt(D^2 - (R1-R2)^2);     % length of each connecting straight
L_arc2     = (pi - 2*alpha)*R2;         % small-circle arc length (right side)
L_arc1     = (pi + 2*alpha)*R1;         % large-circle arc length (left side)
L_total    = 2*L_str + L_arc1 + L_arc2; % total track length

% The 4 connection (tangent) points between sections
P1_bot = [ R1*sin(alpha),     -R1*cos(alpha) ];   % big circle, bottom tangent
P2_bot = [ D + R2*sin(alpha), -R2*cos(alpha) ];   % small circle, bottom tangent
P2_top = [ D + R2*sin(alpha),  R2*cos(alpha) ];   % small circle, top tangent
P1_top = [ R1*sin(alpha),      R1*cos(alpha) ];   % big circle, top tangent

% Bundle geometry for passing to local functions
geom = struct('R1',R1,'R2',R2,'D',D,'alpha',alpha, ...
              'L_str',L_str,'L_arc1',L_arc1,'L_arc2',L_arc2, ...
              'L_total',L_total, ...
              'P1_bot',P1_bot,'P2_bot',P2_bot,'P1_top',P1_top,'P2_top',P2_top);

% Convention for l: +l is to the LEFT of the direction of motion, which
% in this loop corresponds to the interior of the loop (towards the
% centre-line of the two circles) on every section.

% ---- Process noise ------------------------------------------------------
% Longitudinal noise has a NON-ZERO MEAN given by a mean-reverting drift
% toward v_target. The resulting update v_{k+1} = v_k + drift + Gaussian
% simulates a roughly constant cruising speed despite random fluctuations.
v_target = 8;               % Target (cruise) longitudinal speed  [m/s]  (ASSUMED)
tau_v    = 3;               % Mean-reversion time constant        [s]    (ASSUMED)
sigma_v  = 0.25;            % Std of zero-mean longitudinal fluctuation [m/s] (ASSUMED)

% Lateral (Beta) noise design:
%     X ~ Beta( a(l), b(l) ) ,   w_l = beta_scale * ( X - 0.5 )
% Note the fixed shift of 0.5 (NOT a/(a+b)): this gives the noise a NON-ZERO
% MEAN whenever the Beta is skewed (i.e. whenever l != 0). The shape is
% designed so that:
%   - when l = 0    : a = b, the Beta is symmetric about 0.5, so E[w_l] = 0
%   - when l > 0    : a << b, mass of X piles up near 0, so E[w_l] < 0
%   - when l < 0    : a >> b, mass of X piles up near 1, so E[w_l] > 0
% which produces a strong RESTORING FORCE that actively keeps the car near
% the centre line of the track. The skew is deliberately aggressive.
beta_scale  = 0.01;          % m/s scale of the lateral noise        (ASSUMED)
beta_a0     = 10;            % baseline Beta concentration           (ASSUMED)
beta_k_conc = 6;            % concentration drop at |l|=l_max       (ASSUMED)
beta_k_skew = 0.5;         % AGGRESSIVE skew strength (in [0,1))   (ASSUMED)

% ---- Measurement (3 beacons) -------------------------------------------
beacons = [ -40,   40;
             60,  -50;
            110,   35 ];                % beacon positions in world frame [m] (ASSUMED)
Nb      = size(beacons,1);
sigma_r = 0.5;              % range measurement noise std         [m]   (ASSUMED)

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
    % (1a) Longitudinal noise w_v.
    % This has a NON-ZERO MEAN given by the mean-reversion drift
    %      drift_v = (v_target - v) * h/tau_v
    % which pulls the speed back to v_target over a time-scale tau_v.
    % A zero-mean Gaussian fluctuation of std sigma_v is then added.
    % The net effect is a cruise at approximately constant velocity
    % (E[v] -> v_target in steady state) instead of an unbounded
    % random walk.
    drift_v = (v_target - v) * (h/tau_v);
    w_v     = drift_v + sigma_v * randn;

    % (1b) Lateral noise w_l ~ Beta-based, with shape depending on l.
    % Concentration (a+b) is high near the centre (narrow distribution)
    % and drops toward l_max (wider, heavier-tailed distribution).
    % Skew is pushed the OPPOSITE way to l, so:
    %     l > 0  =>  a < b  =>  Beta mass near 0  =>  (X - 0.5) < 0
    %     l < 0  =>  a > b  =>  Beta mass near 1  =>  (X - 0.5) > 0
    % Because we subtract the constant 0.5 (not the true Beta mean),
    % the noise has a non-zero mean whose sign is OPPOSITE to l, i.e.
    % a restoring drift that keeps the car close to the centre line.
    conc = max(beta_a0 - beta_k_conc*(abs(l)/l_max), 1.5);
    skew = -beta_k_skew * (l/l_max);          % l>0 -> skew<0 -> a<b
    a_b  = max(conc*(1 + skew)/2, 0.5);
    b_b  = max(conc*(1 - skew)/2, 0.5);
    Xb   = my_betarnd(a_b, b_b);
    w_l  = beta_scale * ( Xb - 0.5 );         % NON-zero mean: restoring force

    % --------------------------------------------------------------------
    % (2)  STATE UPDATE (forward Euler)
    % --------------------------------------------------------------------
    % Arc-length integrates the longitudinal speed; mod L_total wraps the
    % car around the closed loop so it can lap indefinitely.
    s_new  = mod(s  + h*v,  L_total);

    % Longitudinal speed: previous speed + mean-reverting + fluctuating noise.
    v_new  = v  + w_v;

    % Lateral position: previous + h*lateral_speed.
    l_new  = l  + h*vl;

    % Lateral speed: previous + restoring Beta noise.
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


function X = my_betarnd(a, b)
%MY_BETARND  Beta(a,b) sample without requiring the Statistics Toolbox.
%   Uses the property X = G1/(G1+G2) where Gi ~ Gamma(ai, 1).
    if exist('betarnd','file') == 2
        X = betarnd(a, b);
    else
        g1 = my_gamrnd(a);
        g2 = my_gamrnd(b);
        X  = g1 / (g1 + g2);
    end
end


function g = my_gamrnd(a)
%MY_GAMRND  Gamma(a,1) sample (shape a, unit scale) via Marsaglia-Tsang.
    if a < 1                                 % boost and then shrink
        g = my_gamrnd(a+1) * rand^(1/a);
        return
    end
    d = a - 1/3;
    c = 1/sqrt(9*d);
    while true
        z = randn;
        v = (1 + c*z)^3;
        if v > 0
            u = rand;
            if u < 1 - 0.0331*z^4,                        g = d*v; return; end
            if log(u) < 0.5*z^2 + d*(1 - v + log(v)),     g = d*v; return; end
        end
    end
end


function d = wrap_diff(e, L)
%WRAP_DIFF  Wrap an error signal into (-L/2, L/2] (for a periodic variable of period L).
    d = mod(e + L/2, L) - L/2;
end