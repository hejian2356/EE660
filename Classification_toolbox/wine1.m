clear all
% import data wine
load wine
%mapstd to normalize to 0 mean and 1 variance
train_data = mapstd(feature_train');
test_data = mapstd(feature_test');
 
label_tr = double(label_train)';
wine_test_res_perceptron = multiclass(train_data, label_tr, train_data, '[''OAA'', 0, ''Perceptron'', [2000]]');
fprintf('the right rate of train data for wine with perceptron method is');
disp(mean(wine_test_res_perceptron == label_train'));
 
wine_test_res_perceptron = multiclass(train_data, label_tr, test_data, '[''OAA'', 0, ''Perceptron'', [2000]]');
fprintf('the right rate of test data for wine with perceptron method is');
disp(mean(wine_test_res_perceptron == label_test'));
 
wine_test_res_LS = multiclass(train_data, label_tr, train_data, '[''OAA'', 0, ''LS'', []]');
fprintf('the right rate of train data for wine with LS method is');
disp(mean(wine_test_res_LS == label_train'));
 
wine_test_res_LS = multiclass(train_data, label_tr, test_data, '[''OAA'', 0, ''LS'', []]');
fprintf('the right rate of test data for wine with LS method is');
disp(mean(wine_test_res_LS == label_test'));





