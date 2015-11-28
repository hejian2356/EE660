A = importdata('pre_processed.csv', ',');
b = randperm(653);
testNum = 100;
testX = A(b(1:testNum), 1:47);
testY = A(b(1:testNum), 48);
trainX = A(b(testNum+1:653), 1:47);
trainY = A(b(testNum+1:653), 48);