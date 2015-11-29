sampleWithClass0 = sum(trainY == 0);
sampleWithClass1 = sum(trainY == 1);
if sampleWithClass0 > sampleWithClass1
    rightRate = sum(testY == 0)/length(testY);
else
    rightRate = sum(testY == 1)/length(testY);
end
fprintf('right rate of guessWithLargerPriorProbClassifier on test data is: %f\n', rightRate);
