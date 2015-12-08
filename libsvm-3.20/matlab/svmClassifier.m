%SVM
svmTrainY = trainY;
row1 = length(trainY);
for i = 1 : row1
    if trainY(i) == 0
        svmTrainY(i) = -1;
    end
end
row2 = length(testY);
svmTestY = testY;
for i = 1 : row2
    if testY(i) == 0
        svmTestY(i) = -1;
    end
end
   
fprintf('Without normalization:\n');
% linear
fprintf('linear model:\n');
model = svmtrain(svmTrainY, trainX, '-t 0');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, testX, model);
 
% radial basis
fprintf('rbf kernel, gamma = 0.01:\n');
model = svmtrain(svmTrainY, trainX, '-t 2, -g 0.01');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, testX, model);
 
fprintf('rbf kernel, gamma = 0.1:\n');
model = svmtrain(svmTrainY, trainX, '-t 2, -g 0.1');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, testX, model);
 
fprintf('rbf kernel, gamma = 0.3:\n');
model = svmtrain(svmTrainY, trainX, '-t 2, -g 0.3');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, testX, model);
 
fprintf('rbf kernel, gamma = 1:\n');
model = svmtrain(svmTrainY, trainX, '-t 2, -g 1');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, testX, model);

%do normalization
fprintf('\n\nWith normalization:\n');
normalizeTrainX = (mapstd(trainX'))';
normalizeTestX = (mapstd(testX'))';

% linear
fprintf('linear model:\n');
model = svmtrain(svmTrainY, normalizeTrainX, '-t 0');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, normalizeTestX, model);
 
% radial basis
fprintf('rbf kernel, gamma = 0.01:\n');
model = svmtrain(svmTrainY, normalizeTrainX, '-t 2, -g 0.01');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, normalizeTestX, model);
 
fprintf('rbf kernel, gamma = 0.1:\n');
model = svmtrain(svmTrainY, normalizeTrainX, '-t 2, -g 0.1');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, normalizeTestX, model);
 
fprintf('rbf kernel, gamma = 0.3:\n');
model = svmtrain(svmTrainY, normalizeTrainX, '-t 2, -g 0.3');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, normalizeTestX, model);
 
fprintf('rbf kernel, gamma = 1:\n');
model = svmtrain(svmTrainY, normalizeTrainX, '-t 2, -g 1');
[predicted_label, accuracy, decision_values] = svmpredict(svmTestY, normalizeTestX, model);