l = 0 : 0.05 : 1;
l = l';
 
%stnd
stndXtrain = standardizeCols(trainX);
stndXtest = standardizeCols(testX);
ytrain = trainY;
ytest = testY;
[model, bestParam, mu, se] = fitCv(l, @(stndXtrain, ytrain, l)logregFit(stndXtrain, ytrain, 'lambda', l, 'regType', 'L2'), @logregPredict, @zeroOneLossFn, stndXtrain, ytrain);
yhatTrain = logregPredict(model, stndXtrain);
yhatTest = logregPredict(model, stndXtest);
fprintf('best parameter for stnd data is: %f\n', bestParam);
fprintf('right rate on stnd training data: %f\n', sum(yhatTrain == ytrain)/rows(ytrain));
fprintf('right rate on stnd test data: %f\n', sum(yhatTest == ytest)/rows(ytest));
 
%log
logXtrain = log(trainX+0.1);
logXtest = log(testX+0.1);
[model, bestParam, mu, se] = fitCv(l, @(logXtrain, ytrain, l)logregFit(logXtrain, ytrain, 'lambda', l, 'regType', 'L2'), @logregPredict, @zeroOneLossFn, logXtrain, ytrain);
yhatTrain = logregPredict(model, logXtrain);
yhatTest = logregPredict(model, logXtest);
fprintf('best parameter for log data is: %f\n', bestParam);
fprintf('right rate on log training data: %f\n', sum(yhatTrain == ytrain)/rows(ytrain));
fprintf('right rate on log test data: %f\n', sum(yhatTest == ytest)/rows(ytest));
 
%binarization
[row, col] = size(trainX);
biXtrain = zeros(row, col);
for j = 1 : row
    for i = 1:col
        if (trainX(j, i) > 0)
            biXtrain(j, i) = 1;
        end
    end
end
[row, col] = size(testX);
biXtest = zeros(row, col);
for j = 1 : row
    for i = 1:col
        if (testX(j, i) > 0)
            biXtest(j, i) = 1;
        end
    end
end
[model, bestParam, mu, se] = fitCv(l, @(biXtrain, ytrain, l)logregFit(biXtrain, ytrain, 'lambda', l, 'regType', 'L2'), @logregPredict, @zeroOneLossFn, biXtrain, ytrain);
yhatTrain = logregPredict(model, biXtrain);
yhatTest = logregPredict(model, biXtest);
fprintf('best parameter for binarized data is: %f\n', bestParam);
fprintf('right rate on binarized training data: %f\n', sum(yhatTrain == ytrain)/rows(ytrain));
fprintf('right rate on binarized test data: %f\n', sum(yhatTest == ytest)/rows(ytest));
