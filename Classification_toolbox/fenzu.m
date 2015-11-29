clc;
clear;
[data1 text1] = xlsread('t.xlsx');
[n1,c2]=size(data1);
k1=floor(n1*0.8);
k2=n1-k1;
p1 = randperm(n1);
for i= 1: k1
    for j=1:k2
    train_data(i,:)=data1(p1(:,i),:);
    test_data(j,:)=data1(p1(:,k1+j),:);
    end
end
mean_train=mean(train_data);
mean_test=mean(test_data);
var_train=std(train_data);
var_test=std(test_data);
for i=1:52
feature_train_new(:,i)=(train_data(:,i)-mean_train(i))/var_train(i);
feature_test_new(:,i)=(test_data(:,i)-mean_test(i))/var_test(i);
end
feature_train_new = [feature_train_new(:, 1:28) feature_train_new(:, 30:52)];
feature_test_new= [feature_test_new(:, 1:28) feature_test_new(:, 30:52)];
trainlabel=train_data(:,53);
testlabel=test_data(:,53);
idx1=find(trainlabel==1);
trainlabel(idx1)=0;
idx2=find(trainlabel==2);
trainlabel(idx2)=1;
idx3=find(testlabel==1);
testlabel(idx3)=0;
idx4=find(testlabel==2);
testlabel(idx4)=1;
%perceptron
[p_test,~]= Perceptron(feature_train_new',trainlabel', feature_test_new', 20000);
[m_test,~]=LS(feature_train_new',trainlabel', feature_test_new');
p_test=p_test';
m_test=m_test';
disp(mean(p_test == testlabel))
disp(mean(m_test == testlabel))
