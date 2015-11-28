clc;
allGuessOne = sum(testY)/length(testY);
fprintf('right rate of AllGuessZeroClassifier on test data is: %f\n', 1-allGuessOne);
fprintf('right rate of AllGuessOneClassifier on test data is: %f\n', allGuessOne);
