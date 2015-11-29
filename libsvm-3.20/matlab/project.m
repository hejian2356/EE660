clear all
clc
load train
load test
load train_label
load test_label

train = train_with_id;
test = test_with_id;
[row1, ~] = size(train_label);
train_label_1 = zeros(row1, 1);
for i = 1 : row1
    if train_label(i) == 1
        train_label_1(i) = -1;
    else
        train_label_1(i) = 1;
    end
end
[row2, ~] = size(test_label);
test_label_1 = zeros(row2, 1);
for i = 1 : row2
    if test_label(i) == 1
        test_label_1(i) = -1;
    else
        test_label_1(i) = 1;
    end
end

phi1 = zeros(27439, 1711);
temp = train;
train = [train phi1];
for i = 1 : 27439
    ll = 59;
    for j = 1 : 58
        for k = j : 58
            train(i, ll) = temp(i, j) * temp(i, k);
            ll = ll + 1;
        end
    end
end

phi1 = zeros(3049, 1711);
temp = test;
test = [test  phi1];
for i = 1 : 3049
    ll = 59;
    for j = 1 : 58
        for k = j : 58
            test(i, ll) = temp(i, j) * temp(i, k);
            ll = ll + 1;
        end
    end
end

% linear
model = svmtrain(train_label_1, train, '-t 0');
[predicted_label, accuracy, decision_values] = svmpredict(test_label_1, test, model);

% radial basis, gamma = 10
model = svmtrain(train_label_1, train, '-t 2, -g 10');
[predicted_label, accuracy, decision_values] = svmpredict(test_label_1, test, model);







