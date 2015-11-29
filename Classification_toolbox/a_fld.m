clear all
clc
load train
load test
load train_label
load test_label

train = (train_with_id');
test = (test_with_id');
train_label_1 = train_label' - 1;
test_label_1 = test_label' - 1;

[new_patterns, new_targets, w] = FishersLinearDiscriminant(train, train_label_1);
train = new_patterns(1, :);
train_label = new_targets;
[new_patterns, new_targets, w] = FishersLinearDiscriminant(test, test_label_1);
test = new_patterns(1, :);
test_label = new_targets;

[test_targets, ~] = Perceptron(train, train_label, train, 20000);
fprintf('the right rate of train data with perceptron method is');
disp(mean(test_targets == train_label));

[test_targets, ~] = Perceptron(train, train_label, test, 20000);
fprintf('the right rate of test data with perceptron method is');
disp(mean(test_targets == test_label));




load train
load test
load train_label
load test_label


train = mapstd(train_with_id');
test = mapstd(test_with_id');
train_label_1 = train_label' - 1;
test_label_1 = test_label' - 1;

[test_targets, ~] = Perceptron(train, train_label_1, train, 20000);
fprintf('the right rate of train data with perceptron method is');
disp(mean(test_targets == train_label_1));

[test_targets, ~] = Perceptron(train, train_label_1, test, 20000);
fprintf('the right rate of test data with perceptron method is');
disp(mean(test_targets == test_label_1));