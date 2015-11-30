[n,d] = size(trainX);

nfolds = 5;
[trainfolds, testfolds] = Kfold(n, nfolds,1);

depth = 0:8;
errors = zeros(numel(depth),n);
for f=1:nfolds
     XtrainCV = trainX(trainfolds{f},:); ytrainCV = trainY(trainfolds{f});
     XtestCV = trainX(testfolds{f},:); ytestCV = trainY(testfolds{f});
     for d = 1:numel(depth)
         tree = dtfit(XtrainCV,ytrainCV,'maxdepth',depth(d));
         yhatCV = dtpredict(tree,XtestCV);
         errors(d,testfolds{f}) = (yhatCV-ytestCV).^2;
     end
end
errMean = mean(errors,2);
errStd = std(errors,[],2)/sqrt(n);
bestDepth =  oneStdErrorRule(errMean, errStd);
 
tree = dtfit(trainX,trainY,'maxdepth',bestDepth);
dtdisplay(tree);
yhat = dtpredict(tree,testX);
mseFinal = mse(yhat,testY);
fprintf('right rate of decision tree classifier on test data is: %f\n', sum(yhat == testY)/length(testY));