clear all
clc
load train
load test
load train_label
load test_label

%SVM
train = ((train_with_id'))';
test = ((test_with_id'))';
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
 
model = svmtrain(train_label_1, train, '-t 2, -g 500');
[predicted_label, accuracy, decision_values] = svmpredict(test_label_1, test, model);

