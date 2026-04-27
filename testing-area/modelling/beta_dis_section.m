%% Beta-based Lateral Noise PDF - Visualisation across l
% ------------------------------------------------------------------------
% Standalone script. Shows how the probability density function (PDF) of
% the lateral-noise random variable w_l changes SHAPE as the car's lateral
% offset l is varied from one wall (-l_max) to the other (+l_max), in
% increments of 0.4 m. That gives 11 values of l and therefore 11 PDFs.
% ------------------------------------------------------------------------

clear; clc; close all;

%% ======================================================================
%  PARAMETERS  (same constants used in the main simulation track_sim.m)
% =======================================================================
l_max       = 2.0;     % Track half-width [m]. The car is confined to
                       % |l| <= l_max. Here l_max = 2 m since the track
                       % is 4 m wide.
beta_a0     = 6;       % BASELINE BETA CONCENTRATION at the centre (l = 0).
                       % Concentration = a + b. Larger concentration
                       % means a narrower (more peaked) Beta. With
                       % beta_a0 = 6 and l = 0 the Beta is Beta(3, 3),
                       % a fairly tight symmetric bump around x = 0.5.
beta_k_conc = 4;       % Concentration DROP at |l| = l_max. The actual
                       % concentration used at position l is
                       %     conc(l) = beta_a0 - beta_k_conc * |l|/l_max
                       % so it goes linearly from beta_a0 (centre) to
                       % beta_a0 - beta_k_conc (walls). With these numbers
                       % conc ranges from 6 (centre) down to 2 (walls),
                       % i.e. the Beta gets much wider near the walls.
beta_k_skew = 0.7;     % SKEW STRENGTH at the walls, in [0, 1). Controls
                       % how asymmetric the Beta becomes when the car is
                       % off-centre. skew = 0 means a = b (symmetric).
                       % skew -> +/- 1 means all mass piles near one edge.
beta_scale  = 0.1;     % m/s SCALE. The Beta variable X lives in (0, 1).
                       % The lateral noise w_l is a rescaled, re-centred
                       % version of X with units of m/s. beta_scale sets
                       % the overall magnitude: roughly, |w_l| <= beta_scale/2.

%% ======================================================================
%  BETA-PARAMETER DESIGN
% =======================================================================
% At each value of l, two shape parameters (a, b) are built as:
%
%   conc(l) = max( beta_a0 - beta_k_conc * |l|/l_max ,  1.5 )   % = a + b
%   skew(l) = -beta_k_skew * l/l_max                            % in [-beta_k_skew, +beta_k_skew]
%
%   a(l)    = max( conc*(1 + skew)/2 , 0.5 )                    % mass at x=1 end
%   b(l)    = max( conc*(1 - skew)/2 , 0.5 )                    % mass at x=0 end
%
% Observations:
%   - a + b = conc(l) (up to the 0.5 floors). So CONCENTRATION controls
%     the sum of a and b (i.e. how peaked vs. how spread-out the Beta is).
%   - a - b = conc * skew. So SKEW controls the imbalance between a and b.
%   - At l = 0 we get skew = 0, hence a = b = conc/2 -> symmetric Beta.
%   - At l > 0 we get skew < 0, hence a < b -> Beta mass piles near x = 0.
%   - At l < 0 we get skew > 0, hence a > b -> Beta mass piles near x = 1.
%
% The floors (1.5 on conc, 0.5 on a and b) stop the distribution from
% getting numerically pathological at the extreme walls.
%
% The actual noise used in the simulation is then built as:
%
%     X   ~  Beta(a, b)                        (a dimensionless sample in (0,1))
%     w_l =  beta_scale * ( X - a/(a+b) )      (shifted so E[w_l] = 0, scaled to m/s)
%
% Subtracting a/(a+b) is important: a/(a+b) is the true mean of X, so
% this shift makes E[w_l] = 0 at EVERY value of l, regardless of skew.
% The lateral noise is therefore zero-mean but changes SHAPE with l.

%% ======================================================================
%  EVALUATE AND PLOT THE PDF OF w_l AT 11 VALUES OF l
% =======================================================================
% The PDF of w_l in m/s is obtained from the Beta PDF of X by a simple
% change of variables. With w = beta_scale*(x - mu_X), we have dw/dx =
% beta_scale, so
%
%     pdf_w(w) = pdf_X(x) / beta_scale ,     where x = w/beta_scale + mu_X .
%
% In code: we evaluate pdf_X on a fixed grid of x in (0,1), then map each
% x to its corresponding w, and divide pdf_X by beta_scale.

l_values = -l_max : 0.4 : l_max;            % -2.0, -1.6, -1.2, ..., +1.6, +2.0 (11 values)
x_grid   = linspace(1e-3, 1 - 1e-3, 500);   % avoid 0 and 1 where the PDF may blow up

figure('Position', [80 80 1400 820], 'Name', 'Lateral noise PDF vs l');

for i = 1:numel(l_values)
    l = l_values(i);

    % ---- Beta parameters at this l (the design formulas from above) ----
    conc = max( beta_a0 - beta_k_conc * abs(l)/l_max,  1.5 );
    skew = -beta_k_skew * (l/l_max);
    a    = max( conc*(1 + skew)/2 , 0.5 );
    b    = max( conc*(1 - skew)/2 , 0.5 );

    % ---- Beta PDF of X on (0,1), evaluated directly from the definition
    %      pdf_X(x) = x^(a-1) * (1-x)^(b-1) / B(a, b),  where B is the
    %      built-in beta function (base MATLAB). No Statistics Toolbox
    %      needed.
    pdf_X = x_grid.^(a-1) .* (1 - x_grid).^(b-1) / beta(a, b);

    % ---- Change of variables X -> w_l ----------------------------------
    mu_X   = a / (a + b);                    % mean of Beta(a, b)
    w_axis = beta_scale * (x_grid - mu_X);   % w_l axis
    pdf_w  = pdf_X / beta_scale;             % |dx/dw| = 1/beta_scale

    % ---- Statistics used in the subplot title --------------------------
    std_X = sqrt( a*b / ((a + b)^2 * (a + b + 1)) );   % std of Beta(a, b)
    std_w = beta_scale * std_X;                        % std of w_l

    % ---- Plot -----------------------------------------------------------
    subplot(3, 4, i); hold on; grid on; box on;
    plot(w_axis, pdf_w, 'b-', 'LineWidth', 1.5);
    yL = ylim;                                          % after autoscale
    plot([0 0], [0 yL(2)], 'r-', 'LineWidth', 1.3);     % E[w_l] = 0 reference
    ylim([0 yL(2)]);
    xlim([-beta_scale, beta_scale]);
    xlabel('w_l  [m/s]');
    ylabel('pdf(w_l)');
    title( sprintf('l = %+.1f m    a=%.2f  b=%.2f\nStd[w_l] = %.3f m/s', ...
                    l, a, b, std_w), 'FontSize', 9 );
end

% Use the empty 12th subplot tile as a parameter summary box
subplot(3, 4, 12); axis off;
text(0.02, 0.5, sprintf([ ...
    '  Parameters\n' ...
    '  ----------\n' ...
    '  beta\\_a0     = %g   (centre conc.)\n' ...
    '  beta\\_k\\_conc = %g   (conc. drop)\n' ...
    '  beta\\_k\\_skew = %.2f (skew at wall)\n' ...
    '  beta\\_scale  = %g m/s\n' ...
    '  l\\_max       = %g m\n\n' ...
    '  At each l:\n' ...
    '    conc = beta\\_a0 - beta\\_k\\_conc*|l|/l\\_max\n' ...
    '    skew = -beta\\_k\\_skew*(l/l\\_max)\n' ...
    '    a    = conc*(1+skew)/2\n' ...
    '    b    = conc*(1-skew)/2\n' ...
    '  w\\_l = beta\\_scale * (X - a/(a+b)),\n' ...
    '  X ~ Beta(a, b).'], ...
    beta_a0, beta_k_conc, beta_k_skew, beta_scale, l_max), ...
    'FontSize', 9, 'VerticalAlignment', 'middle', ...
    'FontName', 'Courier New');