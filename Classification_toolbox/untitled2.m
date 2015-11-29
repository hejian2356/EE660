clear all
clc
load train_red
load test_red
load train_label_red
load test_label_red

test_label = test_label_reduction;
train_label = train_label_reduction;

%random guess without prior
count_right = 0;
total_number = size(test_label);
num = total_number(1);
for i = 1 : num
    random_num = randi(1001);
    if (random_num < 500.5 && test_label(i) == 1) || (random_num > 500.5 && test_label(i) == 2)
        count_right = count_right + 1;
    end
end
fprintf('the right rate of test data with random guess without priors is %f\n', double(count_right)/num);

%random guess with prior, assign using random number
count_right = 0;
total_number = size(test_label);
num = total_number(1);
total_train_number = size(train_label);
total_train_number = total_train_number(1);
prior1 = sum(train_label == 1)/double(total_train_number);
threshold = 1000*prior1;
for i = 1 : num
    random_num = randi(1001);
    if (random_num < threshold && test_label(i) == 1) || (random_num >= threshold && test_label(i) == 2)
        count_right = count_right + 1;
    end
end
fprintf('the right rate of test data with random guess with priors and random assignment is %f\n', double(count_right)/num);        

%random guess with prior, assign all to calss with bigger probability
count_right = 0;
total_number = size(test_label);
num = total_number(1);
if sum(train_label == 1) > sum(train_label == 2)
    random_class = 1;
else
    random_class = 2;
end
for i = 1 : num
    if (test_label(i) == random_class)
        count_right = count_right + 1;
    end
end
fprintf('the right rate of test data with random guess with priors and constant assignment is %f\n', double(count_right)/num);   


%base-line: minimum distance to class means classifier
train1 = mapstd(train')';
test1 = mapstd(test')';
total_train_number = size(train_label);
train_num = total_train_number(1);
row1 = sum(train_label == 1);
row2 = sum(train_label == 2);
train_class1 = zeros(row1, 58);
train_class2 = zeros(row2, 58);
j = 1;
k = 1;
for i = 1 : train_num
    if(train_label(i) == 1)
        train_class1(j, :) = train1(i, :);
        j = j + 1;
    else
        train_class2(k, :) = train1(i, :);
        k = k + 1;
    end
end
mean1 = mean(train_class1);
mean2 = mean(train_class2);
count_right = 0;
total_number = size(test_label);
num = total_number(1);
for i = 1 : num
    if (norm(test1(i,:)-mean1) > norm(test1(i,:)-mean2) && test_label(i) == 2) || (norm(test1(i,:)-mean1) <= norm(test1(i,:)-mean2) && test_label(i) == 1)
        count_right = count_right + 1;
    end
end
fprintf('the right rate of test data with minimum distance to class means classifier is %f\n', double(count_right)/num);  


%base-line: perceptron
train1 = mapstd(train');
test1 = mapstd(test');
train_label_1 = train_label_reduction' - 1;
test_label_1 = test_label_reduction' - 1;

[test_targets, ~] = Perceptron(train1, train_label_1, train1, 20000);
fprintf('the right rate of train data with perceptron method is');
disp(mean(test_targets == train_label_1));

[test_targets, ~] = Perceptron(train1, train_label_1, test1, 20000);
fprintf('the right rate of test data with perceptron method is');
disp(mean(test_targets == test_label_1));

%base-line: LS
train1 = mapstd(train');
test1 = mapstd(test');

[test_targets, ~] = LS_modified(train1, train_label_1, train1, []);
fprintf('the right rate of train data with LS method is');
disp(mean(test_targets == train_label_1));

[test_targets, ~] = LS_modified(train1, train_label_1, test1, []);
fprintf('the right rate of test data with LS method is');
disp(mean(test_targets == test_label_1));


%statistical classifier
train1 = mapstd(train');
test1 = mapstd(test');


%KNN
t = Nearest_Neighbor(train1, train_label_1, test1, 20);
fprintf('the right rate of test data with KNN is');
disp(mean(t == test_label_1))

%parzen
t = Parzen(train1, train_label_1, test1, 100);
fprintf('the right rate of test data with parzen is');
disp(mean(t == test_label_1))


