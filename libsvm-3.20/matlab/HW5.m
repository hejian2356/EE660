clear all
load HW5_2
% gamma = 10
model = svmtrain(train_y, train_x, '-t 2, -g 10');
[predicted_label, accuracy, decision_values] = svmpredict(train_y, train_x, model);
plotboundary(train_y, train_x, model, 1)
% gamma = 50
model = svmtrain(train_y, train_x, '-t 2, -g 50');
[predicted_label, accuracy, decision_values] = svmpredict(train_y, train_x, model);
plotboundary(train_y, train_x, model, 1)
% gamma = 5000
model = svmtrain(train_y, train_x, '-t 2, -g 5000');
[predicted_label, accuracy, decision_values] = svmpredict(train_y, train_x, model);
plotboundary(train_y, train_x, model, 1)