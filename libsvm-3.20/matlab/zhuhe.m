clc;
clear;
load('HW5_1');
model1=svmtrain(train_y,train_x,'-t 0 -c 1');
model2=svmtrain(train_y,train_x,'-t 0 -c 100');
plotboundary(train_y,train_x,model1);
plotboundary(train_y,train_x,model2);
a=model2.SVs
c=model2.sv_coef;
b=model2.rho
w=c'*a;
g=a*w'-b;

