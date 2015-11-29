clear all
clc
load train_red
load test_red
load train_label_red
load test_label_red

%expansion
train = mapstd(train');
test = mapstd(test');
train_label_1 = train_label_reduction' - 1;
test_label_1 = test_label_reduction' - 1;

[new_patterns, new_targets] = PCA(train, train_label_1, 5);
train = new_patterns;
train_label_1 = new_targets;
[new_patterns, new_targets] = PCA(test, test_label_1, 5);
test = new_patterns;
test_label_1 = new_targets;

[test_targets, ~] = Perceptron(train, train_label_1, train, 20000);
fprintf('after PCA, the right rate of train data with perceptron method is');
disp(mean(test_targets == train_label_1));

[test_targets, ~] = Perceptron(train, train_label_1, test, 20000);
fprintf('after PCA, the right rate of test data with perceptron method is');
disp(mean(test_targets == test_label_1));

phi1 = zeros(15, 2743);
temp = train;
train = [train; phi1];
for i = 1 : 2743
    ll = 6;
    for j = 1 : 5
        for k = j : 5
            train(ll, i) = temp(j, i) * temp(k, i);
            ll = ll + 1;
        end
    end
end

phi1 = zeros(15, 304);
temp = test;
test = [test; phi1];
for i = 1 : 304
    ll = 6;
    for j = 1 : 5
        for k = j : 5
            test(ll, i) = temp(j, i) * temp(k, i);
            ll = ll + 1;
        end
    end
end




[test_targets, ~] = Perceptron(train, train_label_1, train, 20000);
fprintf('after quadratic expansion, the right rate of train data with perceptron method is');
disp(mean(test_targets == train_label_1));

[test_targets, ~] = Perceptron(train, train_label_1, test, 20000);
fprintf('after quadratic expansion, the right rate of test data with perceptron method is');
disp(mean(test_targets == test_label_1));




