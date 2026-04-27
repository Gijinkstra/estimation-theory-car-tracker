clear; clc; close all; rng(42);

%% ====================== SIMULATION SETUP =========================
h = 0.1;
T_sim = 500;
N = round(T_sim/h);

R1=50; R2=20; D=100;
track_w=4; l_max=track_w/2;

alpha = asin((R1-R2)/D);
L_str = sqrt(D^2 - (R1-R2)^2);
L_arc2 = (pi - 2*alpha)*R2;
L_arc1 = (pi + 2*alpha)*R1;
L_total = 2*L_str + L_arc1 + L_arc2;

P1_bot=[ R1*sin(alpha), -R1*cos(alpha)];
P2_bot=[ D+R2*sin(alpha), -R2*cos(alpha)];
P2_top=[ D+R2*sin(alpha),  R2*cos(alpha)];
P1_top=[ R1*sin(alpha),   R1*cos(alpha)];

geom = struct('R1',R1,'R2',R2,'D',D,'alpha',alpha,...
    'L_str',L_str,'L_arc1',L_arc1,'L_arc2',L_arc2,...
    'L_total',L_total,...
    'P1_bot',P1_bot,'P2_bot',P2_bot,'P1_top',P1_top,'P2_top',P2_top);

sigma_v=0.25;

beta_scale=0.1;
beta_a0=10;
beta_k_conc=6;
beta_k_skew=0.5;

beacons=[-40 40;60 -50;110 35];
Nb=size(beacons,1);
sigma_r=0.5;

x0=[0;8;0;0];

%% ====================== STORAGE =========================
X=zeros(4,N+1); X(:,1)=x0;
XY=zeros(2,N+1);

% NEW: store beta parameters
A_beta=zeros(1,N);
B_beta=zeros(1,N);

[XY(1,1),XY(2,1)] = sl_to_xy(X(1,1),X(3,1),geom);

%% ====================== SIM LOOP =========================
for k=1:N

    s=X(1,k); v=X(2,k); l=X(3,k); vl=X(4,k);

    w_v = sigma_v*randn;

    % --- Beta parameters ---
    conc = max(beta_a0 - beta_k_conc*(abs(l)/l_max),1.5);
    skew = -beta_k_skew*(l/l_max);
    a_b  = max(conc*(1+skew)/2,0.5);
    b_b  = max(conc*(1-skew)/2,0.5);

    % STORE THEM
    A_beta(k)=a_b;
    B_beta(k)=b_b;

    % sample
    Xb = my_betarnd(a_b,b_b);
    w_l = beta_scale*(Xb-0.5);

    % state update
    s_new = mod(s + h*v, L_total);
    v_new = v + w_v;
    l_new = l + h*vl;
    vl_new = vl + w_l;

    if l_new>l_max, l_new=l_max; vl_new=-0.2*vl_new; end
    if l_new<-l_max, l_new=-l_max; vl_new=-0.2*vl_new; end

    X(:,k+1)=[s_new;v_new;l_new;vl_new];

    [xw,yw]=sl_to_xy(s_new,l_new,geom);
    XY(:,k+1)=[xw;yw];
end

%% ====================== PREP TRACK =========================
s_grid = linspace(0,L_total,800);
xy_c=zeros(2,numel(s_grid));
for i=1:numel(s_grid)
    [xy_c(1,i),xy_c(2,i)] = sl_to_xy(s_grid(i),0,geom);
end

%% ====================== ANIMATION =========================
figure;

% --- Track subplot ---
subplot(1,2,1); hold on; axis equal; grid on;
plot(xy_c(1,:),xy_c(2,:),'k--');

h_car = plot(XY(1,1),XY(2,1),'bo','MarkerFaceColor','b');

title('Vehicle on Track');

% --- Beta PDF subplot ---
subplot(1,2,2); hold on; grid on;
x_pdf = linspace(0,1,200);
h_pdf = plot(x_pdf, zeros(size(x_pdf)),'r','LineWidth',1.5);

title('Beta Distribution (noise)');
xlabel('x'); ylabel('pdf');

%% ====================== ANIMATION =========================
figure;

t = (0:N)*h;

% --- (1) Track subplot ---
subplot(1,3,1); hold on; axis equal; grid on;
plot(xy_c(1,:),xy_c(2,:),'k--');
h_car = plot(XY(1,1),XY(2,1),'bo','MarkerFaceColor','b');
title('Vehicle on Track');

% --- (2) Beta PDF subplot ---
subplot(1,3,2); hold on; grid on;
x_pdf = linspace(0,1,200);
h_pdf = plot(x_pdf, zeros(size(x_pdf)),'r','LineWidth',1.5);
title('Beta Distribution');
xlabel('x'); ylabel('pdf');

% --- (3) Lateral position subplot ---
subplot(1,3,3); hold on; grid on;
plot(t, X(3,:), 'b-', 'LineWidth',1);   % full trajectory

h_l_point = plot(t(1), X(3,1), 'ro','MarkerFaceColor','r'); % moving point
h_l_line  = xline(t(1),'r--');  % moving vertical time cursor

xlabel('time [s]');
ylabel('l [m]');
title('Lateral Position');

ylim([-l_max l_max]);

%% ====================== RUN ANIMATION =========================
for k = 1:5:N

    % --- Vehicle position ---
    set(h_car,'XData',XY(1,k),'YData',XY(2,k));

    % --- Beta PDF ---
    a = A_beta(k);
    b = B_beta(k);

    Bfunc = gamma(a)*gamma(b)/gamma(a+b);
    pdf = (x_pdf.^(a-1) .* (1-x_pdf).^(b-1)) / Bfunc;

    set(h_pdf,'YData',pdf);
    title(sprintf('Beta PDF (a=%.2f, b=%.2f)',a,b));

    % --- Lateral state update ---
    set(h_l_point,'XData',t(k),'YData',X(3,k));
    set(h_l_line,'Value',t(k));

    drawnow;
    pause(0.05);
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