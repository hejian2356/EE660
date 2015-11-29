%MSE classifier
[test_targets, ~] = LS_modified(trainX', trainY', trainX', []);
rightRate = sum(test_targets == trainY')/length(trainY);
fprintf('Without normalization:\n');
fprintf('right rate of perceptron on training data is: %f\n', rightRate);
 
[test_targets, ~] = LS_modified(trainX', trainY', testX', []);
rightRate = sum(test_targets == testY')/length(testY);
fprintf('right rate of perceptron on training data is: %f\n', rightRate);

%do normalization
normalizeTrainX = mapstd(trainX');
normalizeTestX = mapstd(testX');
[test_targets, ~] = LS_modified(normalizeTrainX, trainY', normalizeTrainX, []);
rightRate = sum(test_targets == trainY')/length(trainY);
fprintf('With normalization:\n');
fprintf('right rate of perceptron on training data is: %f\n', rightRate);

[test_targets, ~] = LS_modified(normalizeTrainX, trainY', normalizeTestX, []);
rightRate = sum(test_targets == testY')/length(testY);
fprintf('right rate of perceptron on test data is: %f\n', rightRate);