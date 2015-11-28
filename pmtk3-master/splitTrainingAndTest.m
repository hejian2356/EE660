A = importdata('pre_processed.csv', ',');
b = randperm(653);
testX = A(b(1:65), 1:47);
testY = A(b(1:65), 48);
trainX = A(b(66:653), 1:47);
trainY = A(b(66:653), 48);