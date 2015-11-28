clc;
predict_Y = rand(length(testY), 1);
predict_Y(predict_Y > 0.5) = 1;
predict_Y(predict_Y <= 0.5) = 0;
fprintf('right rate on test data is: %f\n', sum(predict_Y == testY)/length(testY));