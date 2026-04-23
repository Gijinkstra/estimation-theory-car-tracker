%% Lateral Beta Noise - 3-scenario Calibration via Interpolation
% ------------------------------------------------------------------------
% Instead of designing (a, b) from a closed-form formula in l, we pick
% (a, b) DIRECTLY at three anchor points in l:
%
%     l = -l_max      (car @ LEFT wall)
%     l =  0          (car @ centre)
%     l = +l_max      (car @ RIGHT wall)
%
% then linearly interpolate a(l) and b(l) between them, and linearly
% extrapolate outside [-l_max, +l_max]. The Beta PDF of X on (0,1)
% is built from (a(l), b(l)), and w_l is the shifted-and-scaled version
% with zero mean.
% ------------------------------------------------------------------------

clear; clc; close all;

%% ======================================================================
%  SHARED PARAMETERS
% =======================================================================
l_max      = 2.0;     % Track half-width [m]. The car is confined to |l| <= l_max.
l_values   = -l_max : 0.4 : l_max;                % 11 values used for the sweep

%% ======================================================================
%  THREE CALIBRATION SCENARIOS
% =======================================================================
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
%   at l = -l_max :  a=5, b=1   (strongly skewed toward x=1)
%   at l =  0     :  a=5, b=5   (symmetric)
%   at l = +l_max :  a=1, b=5   (strongly skewed toward x=0)

l_cal = [ -l_max,    0,     +l_max ];      % anchor l values: -2, 0, +2
a_cal = [  3.0,      3.0,    1.0   ];      % Beta 'a' at each anchor
b_cal = [  1.0,      3.0,    3.0   ];      % Beta 'b' at each anchor

%% ======================================================================
%  INTERPOLATED a(l) AND b(l)
% =======================================================================
% The anchors span the FULL admissible range [-l_max, +l_max]
% Piecewise-linear interpolation blends between the bracketing anchor pair. 

a_of_l = @(l) max( interp1(l_cal, a_cal, l, 'linear'));
b_of_l = @(l) max( interp1(l_cal, b_cal, l, 'linear'));

%% ======================================================================
%  PDF OF w_l AT ANY l
% =======================================================================
% X ~ Beta(a, b),  w_l = beta_scale * ( X - a/(a+b) )
% The subtraction by a/(a+b) uses the Beta's own mean, so E[w_l] = 0 at
% every l. Change of variables: pdf_w(w) = pdf_X(x) / beta_scale.
% The Beta PDF itself is
%     pdf_X(x; a,b) = x^(a-1) * (1-x)^(b-1) / B(a, b)
% using the built-in base-MATLAB beta() function as the normaliser. as per
% the standard way to build up a beta distribution. 

beta_pdf = @(x, a, b) x.^(a-1) .* (1 - x).^(b-1) / beta(a, b);

% ----------------------------------------------------------------------
% Shared axis limits for every plot in the figures below.
% ----------------------------------------------------------------------

xlim_common = [-1, 1];
ylim_common = [0, 3.5];

%% ======================================================================
%  FIGURE 1: the 3 CALIBRATION scenarios overlaid
% =======================================================================
figure('Position', [80 80 780 580], 'Name', 'Calibration scenarios');
hold on; grid on; box on;
cal_colors = [0.20 0.35 0.90;    % blue: left-wall scenario
              0.10 0.10 0.10;    % black: centre
              0.90 0.20 0.20];   % red:  right-wall scenario

x_grid   = linspace(0, 1, 500);   
cal_labels = cell(1, numel(l_cal));
for i = 1:numel(l_cal)
    a = a_cal(i);
    b = b_cal(i);
    mu_X  = a/(a+b);
    w     = (x_grid - mu_X);
    pdf_w = beta_pdf(x_grid, a, b);
    plot(w, pdf_w, 'Color', cal_colors(i,:), 'LineWidth', 2.2);
    cal_labels{i} = sprintf('l = %+.1f m   (a=%.1f, b=%.1f)', ...
                             l_cal(i), a, b);
end
plot([0 0], ylim_common, 'k--', 'LineWidth', 0.8);   % E[w_l] = 0 reference
xlim(xlim_common);
ylim(ylim_common);
xlabel('w_l  [m/s]'); ylabel('pdf(w_l)');
title('Three calibration scenarios');
legend(cal_labels, 'Location', 'northeast');

%% ======================================================================
%  FIGURE 2: all 11 PDFs across the sweep -l_max : 0.4 : +l_max
% =======================================================================
% 11 subplots in a 3x4 grid (one cell left blank). Calibration-anchor
% l values are flagged in the subplot title with "(cal)" so you can
% still tell them apart from the interpolated ones.
% All subplots share identical x and y limits (xlim_common, ylim_common)
% so the shapes can be compared directly by eye.

figure('Position', [80 80 1500 820], 'Name', 'PDFs across full l sweep');

for i = 1:numel(l_values)
    l = l_values(i);
    a = a_of_l(l);
    b = b_of_l(l);
    mu_X  = a/(a+b);
    w     = (x_grid - mu_X);
    pdf_w = beta_pdf(x_grid, a, b);
    std_w = sqrt( a*b / ((a+b)^2 * (a+b+1)) );

    is_anchor = any(abs(l_cal - l) < 1e-9);
    cal_tag = '';
    if is_anchor, cal_tag = '  (cal)'; end

    subplot(3, 4, i); hold on; grid on; box on;
    plot(w, pdf_w, 'b-', 'LineWidth', 1.5);
    plot([0 0], ylim_common, 'r-', 'LineWidth', 1.2);   % E[w_l] = 0 reference
    xlim(xlim_common);
    ylim(ylim_common);
    xlabel('w_l  [m/s]'); ylabel('pdf(w_l)');
    title(sprintf('l = %+.1f m%s    a=%.2f, b=%.2f\nStd[w_l] = %.3f m/s', ...
                  l, cal_tag, a, b, std_w), 'FontSize', 9);
end
