load('spamData.mat')
l = 0 : 0.05 : 2;
l = l';

[row, col] = size(Xtrain);
biXtrain = zeros(row, col);
for j = 1 : row
    for i = 1:col
        if (Xtrain(j, i) > 0)
            biXtrain(j, i) = 1;
        end
    end
end
[row, col] = size(Xtest);
biXtest = zeros(row, col);
for j = 1 : row
    for i = 1:col
        if (Xtest(j, i) > 0)
            biXtest(j, i) = 1;
        end
    end
end
[model, bestParam, mu, se] = fitCv(l, @(biXtrain, ytrain, l)logregFit(biXtrain, ytrain, 'lambda', l, 'regType', 'L2'), @logregPredict, @zeroOneLossFn, biXtrain, ytrain);
yhatTest = logregPredict(model, biXtest);
  
biXtestSum = zeros(row, 2);
for j = 1 : row
    biXtestSum(j, 1) = sum(Xtest(j, 1:48));
    biXtestSum(j, 2) = sum(Xtest(j, 49:54));
end
figure(1);
scatter(biXtestSum(yhatTest == 0 ,1), biXtestSum(yhatTest == 0, 2), 'r');
hold on;
scatter(biXtestSum(yhatTest == 1 ,1), biXtestSum(yhatTest == 1, 2), 'g');
figure(2);
biXtestSumSpam = biXtestSum(yhatTest == 1, :);
hist3(biXtestSumSpam);
figure(3);
biXtestSumNotSpam = biXtestSum(yhatTest == 0, :);
hist3(biXtestSumNotSpam);