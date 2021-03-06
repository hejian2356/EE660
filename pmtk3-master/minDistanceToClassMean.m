class0 = trainX(trainY == 0, :);
class1 = trainX(trainY == 1, :);
mean_0 = mean(class0);
mean_1 = mean(class1);
predict_Y = zeros(length(testY), 1);
for i = 1 : length(testY)
    dis0 = norm(testX(i, :)-mean_0);
    dis1 = norm(testX(i, :)-mean_1);
    if (dis1 < dis0)
        predict_Y(i) = 1;
    end
end
fprintf('Without normalization:\n');
fprintf('right rate on test data is %f\n', sum(predict_Y == testY)/length(testY));

%do normalization
normalizeTrainX = (mapstd(trainX'))';
normalizeTestX = (mapstd(testX'))';
class0 = normalizeTrainX(trainY == 0, :);
class1 = normalizeTrainX(trainY == 1, :);
mean_0 = mean(class0);
mean_1 = mean(class1);
predict_Y = zeros(length(testY), 1);
for i = 1 : length(testY)
    dis0 = norm(normalizeTestX(i, :)-mean_0);
    dis1 = norm(normalizeTestX(i, :)-mean_1);
    if (dis1 < dis0)
        predict_Y(i) = 1;
    end
end
fprintf('With normalization:\n');
fprintf('right rate on test data is %f\n', sum(predict_Y == testY)/length(testY));
