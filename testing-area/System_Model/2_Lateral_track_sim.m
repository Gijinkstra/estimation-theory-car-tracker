clc; clear; close all;

% This follows on from the beta distibution script, this script just simulates a number of runs of the car along a straight line to see how the lateral movement is

%% ================================
% Simulation parameters
% ================================
dt = 0.01;
T  = 20;
N  = T/dt;

%% ================================
% Track
% ================================
L_track = 1000; % increase this to see longer simulations - more road
l_max   = 2;

%% ================================
% Beta interpolation setup
% ================================
l_nodes = [-l_max, 0, +l_max];

% adjust these beta parameters to see different behaviour
a_nodes = [1.0, 50, 5];
b_nodes = [5, 50, 1.0];

sigma_l = 0.1;   %essentially the noise scaling factor

%% ================================
% Monte Carlo runs
% ================================
n_runs = 5;   %increase this to see more simulations

l_all = zeros(n_runs, N);

%% ================================
% Simulation loop (multi-run)
% ================================
for run = 1:n_runs

    % --- State ---
    s  = zeros(1,N);
    v  = zeros(1,N);
    l  = zeros(1,N);
    vl = zeros(1,N);

    % Initial conditions
    s(1)  = 0;
    v(1)  = 5;
    l(1)  = 0;
    vl(1) = 0;

    for k = 1:N-1

        %% Longitudinal
        s(k+1) = s(k) + v(k)*dt;
        if s(k+1) > L_track
            s(k+1) = s(k+1) - L_track;
        end

        %% --- Interpolated Beta (manual) ---
        a_l = interp1(l_nodes, a_nodes, l(k), 'linear', 'extrap');
        b_l = interp1(l_nodes, b_nodes, l(k), 'linear', 'extrap');

        % Gamma samples
        g1 = gamma_rand(a_l);
        g2 = gamma_rand(b_l);

        % Beta sample
        x_beta = g1 / (g1 + g2);

        % Zero-mean noise
        mu_beta = a_l / (a_l + b_l);
        w_l = (x_beta - mu_beta);
        w_l = sigma_l * w_l;

        %% Lateral dynamics
        vl(k+1) = vl(k) + w_l;
        l(k+1)  = l(k) + vl(k)*dt;

        % Boundaries
        if l(k+1) > l_max
            l(k+1) = l_max;
            vl(k+1) = -0.5 * vl(k+1);
        elseif l(k+1) < -l_max
            l(k+1) = -l_max;
            vl(k+1) = -0.5 * vl(k+1);
        end

    end

    l_all(run,:) = l;

end

%% ================================
% Plot trajectories
% ================================
figure('Name','20 Lateral Trajectories','Position',[100 100 900 500]);
hold on; grid on; box on;

for i = 1:n_runs
    plot(l_all(i,:), 'LineWidth',1);
end

yline(l_max,'r--');
yline(-l_max,'r--');

title('Monte Carlo: Lateral Trajectories');
xlabel('Time step');
ylabel('l (lateral position)');

%% ================================
% Gamma random generator
% ================================
function g = gamma_rand(k)

    if k < 1
        u = rand;
        g = gamma_rand(k+1) * u^(1/k);
        return;
    end

    % Marsaglia–Tsang method
    d = k - 1/3;
    c = 1 / sqrt(9*d);

    while true
        x = randn;
        v = (1 + c*x)^3;

        if v <= 0
            continue;
        end

        u = rand;

        if u < 1 - 0.0331*(x^4)
            g = d * v;
            return;
        end

        if log(u) < 0.5*x^2 + d*(1 - v + log(v))
            g = d * v;
            return;
        end
    end
end