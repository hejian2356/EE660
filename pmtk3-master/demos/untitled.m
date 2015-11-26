pointx1 = [1 3.25 3.25^2 3.25^3]';
pointy1 = w'*pointx1;
pointx1 = [1 9.75 9.75^2 9.75^3]';
pointy2 = w'*pointx1;
fprintf('the fitted curve for x = 3.25 is %f\n', pointy1);
fprintf('the fitted curve for x = 9.75 is %f\n', pointy2);