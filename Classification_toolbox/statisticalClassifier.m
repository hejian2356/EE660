%KNN
neighborNum = [3 10 20 50 100];
fprintf('Without normalization:\n');
for k = 1 : length(neighborNum)
    t = Nearest_Neighbor(trainX', trainY', testX', neighborNum(k));
    rightRate = sum(t == testY')/length(testY);
    fprintf('right rate of KNN(%d) on test data is: %f\n', neighborNum(k), rightRate);
end


%Parzen window
windowLenght = [5 10 50 100 200];
for k = 1 : length(windowLenght)
    t = Parzen(trainX', trainY', testX', windowLenght(k));
    rightRate = sum(t == testY')/length(testY);
    fprintf('right rate of Parzen window(%d) on test data is: %f\n', windowLenght(k), rightRate);
end

%do normalization
normalizeTrainX = mapstd(trainX');
normalizeTestX = mapstd(testX');

%KNN
fprintf('With normalization:\n');
for k = 1 : length(neighborNum)
    t = Nearest_Neighbor(normalizeTrainX, trainY', normalizeTestX, neighborNum(k));
    rightRate = sum(t == testY')/length(testY);
    fprintf('right rate of KNN(%d) on test data is: %f\n', neighborNum(k), rightRate);
end

%Parzen window
for k = 1 : length(windowLenght)
    t = Parzen(normalizeTrainX, trainY', normalizeTestX, windowLenght(k));
    rightRate = sum(t == testY')/length(testY);
    fprintf('right rate of Parzen window(%d) on test data is: %f\n', windowLenght(k), rightRate);
end