clear all
% import data wine
load mnist
%mapstd to normalize to 0 mean and 1 variance
train_data = mapstd(feature_train_01');
test_data = mapstd(feature_test_01');
 
label_tr = double(label_train_01)';
wine_test_res_perceptron = multiclass(train_data, label_tr, train_data, '[''all-pairs'', 0, ''Perceptron'', [2000]]');
fprintf('the right rate of train data for mnist with perceptron method is');
disp(mean(wine_test_res_perceptron == label_train_01'));
 
wine_test_res_perceptron = multiclass(train_data, label_tr, test_data, '[''all-pairs'', 0, ''Perceptron'', [2000]]');
fprintf('the right rate of test data for mnist with perceptron method is');
disp(mean(wine_test_res_perceptron == label_test_01'));
 
wine_test_res_LS = multiclass(train_data, label_tr, train_data, '[''all-pairs'', 0, ''LS'', []]');
fprintf('the right rate of train data for mnist with LS method is');
disp(mean(wine_test_res_LS == label_train_01'));
 
wine_test_res_LS = multiclass(train_data, label_tr, test_data, '[''all-pairs'', 0, ''LS'', []]');
fprintf('the right rate of test data for mnist with LS method is');
disp(mean(wine_test_res_LS == label_test_01'));



