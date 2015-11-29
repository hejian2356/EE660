[test_targets, ~] = Perceptron(trainX', trainY', trainX', 5000);
rightRate = sum(test_targets == trainY')/length(trainY);
fprintf('Without normalization:\n');
fprintf('right rate of perceptron on training data is: %f\n', rightRate);

[test_targets, ~] = Perceptron(trainX', trainY', testX', 5000);
rightRate = sum(test_targets == testY')/length(testY);
fprintf('right rate of perceptron on test data is: %f\n', rightRate);

%do normalization
normalizeTrainX = mapstd(trainX');
normalizeTestX = mapstd(testX');
[test_targets, ~] = Perceptron(normalizeTrainX, trainY', normalizeTrainX, 5000);
rightRate = sum(test_targets == trainY')/length(trainY);
fprintf('With normalization:\n');
fprintf('right rate of perceptron on training data is: %f\n', rightRate);

[test_targets, ~] = Perceptron(normalizeTrainX, trainY', normalizeTestX, 5000);
rightRate = sum(test_targets == testY')/length(testY);
fprintf('right rate of perceptron on test data is: %f\n', rightRate);