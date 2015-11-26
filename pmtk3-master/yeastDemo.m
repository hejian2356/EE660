%% preprocessing
clear all;
load yeast;
%{
classes = unique(Y);
for i = 1:size(classes)
	Y(strcmp(classes{i}, Y))={i};
end
%%
y = zeros(size(Y));
for i = 1:size(y)
	y(i) = Y{i};
end
%%
split = 1000;
Xtrain = X(1:split,:);
Ytrain = y(1:split);

Xtest = X(split+1:end,:);
Ytest = y(split+1:end);

%}
%%
nitr = 10;
nntree = 10;
train = [Xtrain Ytrain];
sizeTrain = size(train);
errs_test = zeros(nitr,nntree);
errs_train = zeros(nitr,nntree);
for ntree = 1:1:nntree
	ntree
	for itr = 1:nitr
        sampleRows = randperm(sizeTrain(1));
        sampleRows = sampleRows(1: 100);
        XtrainSample = train(sampleRows, 1: 8);
        YtrainSample = train(sampleRows, 9);
		forest = fitForest(XtrainSample,YtrainSample,'randomFeatures',3,'bagSize',1/2,'ntrees',ntree);
		yhat_test = predictForest(forest,Xtest);
		errs_test(itr,ntree) = mean(Ytest ~= yhat_test);
		yhat_train = predictForest(forest,XtrainSample);
		errs_train(itr,ntree) = mean(YtrainSample ~= yhat_train);
	end
end
disp('finished');
%%
std_vs_ntree = std(errs_test,1);
mean_vs_ntree_test = mean(errs_test,1);
mean_vs_ntree_train = mean(errs_train,1);

plot(std_vs_ntree,'g*');
hold on;
plot(mean_vs_ntree_test,'r*');
hold on;
plot(mean_vs_ntree_train,'*');

title('mean and std of error rates of random forest');
legend('std of test error','mean test error', 'mean train error');
xlabel('Number of Trees');