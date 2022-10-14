theta = [0:pi/180:pi];
x1 = 10;
y1 = 10;
x2 = 15;
y2 = 15;
x3 = 30;
y3 = 30;
hold on
plot(theta, x1*cos(theta) + y1*sin(theta))
plot(theta, x2*cos(theta) + y2*sin(theta))
plot(theta, x3*cos(theta) + y3*sin(theta))
hold off
xlabel('Angle (rad)')
ylabel('Rho Distance (px)')
title('Parametrization for points (10,10), (15,15) and (30, 30)')
legend('(10,10)', '(15,15)', '(30,30)')