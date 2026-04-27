% This bit is Sonnet 4.6 generated code. A lot better than what I had
% written, obviously. This was generated as obviously there is a
% discontinuity between using half a circle arc and drawing a straight
% lines between the two arcs. The method used here is referred to as
% circle-circle tangents. Can read more here: https://mathworld.wolfram.com/Circle-CircleTangents.html
% This would take me ages to implement manually probably.
nSamples = 500;

% Circle definitions
C1 = [0, 0];
C2 = [100, 0];
r1 = 46;
r2 = 20;
r3 = 50;
r4 = 24;

[innerTrackX, innerTrackY] = constructTrack(C1, C2, r1, r2, nSamples);
[outerTrackX, outerTrackY] = constructTrack(C1, C2, r3, r4, nSamples);
midTrackX = (innerTrackX + outerTrackX) ./ 2;
midTrackY = (innerTrackY + outerTrackY) ./ 2;

%% STEP 6: Plot
figure; hold on; axis equal; grid on;
set(gcf, 'Color', 'k');
set(gca, 'Color', 'k');

plot(innerTrackX, innerTrackY, 'b', 'LineWidth', 2, 'DisplayName', 'Inner track')
plot(outerTrackX, outerTrackY, 'r', 'LineWidth', 2, 'DisplayName', 'Outer track')
plot(midTrackX, midTrackY, '--w', 'LineWidth', 2, 'DisplayName', 'Middle track')
legend

title('Racetrack Geometry', 'Color', 'w');
xlabel('X', 'Color', 'w'); ylabel('Y', 'Color', 'w');
T = table(innerTrackX, innerTrackY, outerTrackX, outerTrackY, midTrackX, midTrackY);
T.Properties.VariableNames = {'Inner Track (x)', 'Inner Track (y)', ...
                              'Outer Track (x)', 'Outer Track (y)', ...
                              'Mid Track (x)', 'Mid track (y)'};
T
% writetable(T, "track-coordinates.csv")

function [trackX, trackY] = constructTrack(C1, C2, r1, r2, nSamples)

%% STEP 1: Compute external tangent points
d    = C2 - C1;
dist = norm(d);
u    = d / dist;
perp = [-u(2), u(1)];

alpha = (r1 - r2) / dist;
if abs(alpha) > 1
    error('No external tangents exist');
end
beta = sqrt(1 - alpha^2);

dir1 = alpha * u + beta * perp;
dir2 = alpha * u - beta * perp;

% Tangent points
p1_top = C1 + r1 * dir1;
p2_top = C2 + r2 * dir1;
p1_bot = C1 + r1 * dir2;
p2_bot = C2 + r2 * dir2;

%% STEP 2: Compute tangent angles at each circle
ang1_top = atan2(p1_top(2)-C1(2), p1_top(1)-C1(1));
ang1_bot = atan2(p1_bot(2)-C1(2), p1_bot(1)-C1(1));
ang2_top = atan2(p2_top(2)-C2(2), p2_top(1)-C2(1));
ang2_bot = atan2(p2_bot(2)-C2(2), p2_bot(1)-C2(1));

%% STEP 3: Directly parameterise arcs between tangent points
% Arc on C1: go from bot tangent to top tangent (clockwise)
arc1 = directArc(C1, r1, ang1_bot, ang1_top, nSamples, 'cw');

% Arc on C2: go from top tangent to bot tangent (clockwise)
arc2 = directArc(C2, r2, ang2_top, ang2_bot, nSamples, 'cw');

%% STEP 4: Straight tangent lines
t       = linspace(0, 1, 200)';
topLine = p1_top + (p2_top - p1_top) .* t;
botLine = p2_bot + (p1_bot - p2_bot) .* t;

%% STEP 5: Assemble track (closed loop)
trackX = [arc1(:,1); topLine(:,1); arc2(:,1); botLine(:,1)];
trackY = [arc1(:,2); topLine(:,2); arc2(:,2); botLine(:,2)];

end

function pts = directArc(C, r, angStart, angEnd, n, dir)
    % Ensure angular travel is in the correct direction
    if strcmp(dir, 'ccw')
        if angEnd <= angStart
            angEnd = angEnd + 2*pi;
        end
    else  % cw
        if angEnd >= angStart
            angEnd = angEnd - 2*pi;
        end
    end
    theta = linspace(angStart, angEnd, n);
    pts   = [C(1) + r*cos(theta(:)), C(2) + r*sin(theta(:))];
end