clear all
clc
load train
load test
load train_label
load test_label

%statistical classifier
train = (train_with_id');
test = (test_with_id');
train_label_1 = train_label' - 1;
test_label_1 = test_label' - 1;

%KNN
t = Nearest_Neighbor(train, train_label_1, test, 20);
fprintf('the right rate of test data with KNN (100) is');
disp(mean(t == test_label_1))


%parzen
t = Parzen(train, train_label_1, test, 50);
fprintf('the right rate of test data with parzen (200) is');
disp(mean(t == test_label_1))


