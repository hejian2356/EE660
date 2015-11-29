clear all;
load('mnist.mat')
feature_train=feature_train_01;
feature_test=feature_test_01;
label_train=label_train_01;
label_test=label_test_01;
%perceptron
t = Perceptron(feature_train', double(label_train)', feature_train',5000);
train_error_rate=mean(t ~= label_train')
t = Perceptron(feature_train', double(label_train)', feature_test', 5000);
test_error_rate=mean(t ~= label_test')
%MSE
t = LS_modified(feature_train', double(label_train)', feature_train');
train_error_rate=mean(t ~= label_train')
t = LS_modified(feature_train', double(label_train)', feature_test');
test_error_rate=mean(t ~= label_test')
i=find(label_train==0);
img=reshape(feature_train(i(1),:),28,28);
figure
subplot(2, 2, 1)
imshow(img)
title('1st training image in class 0')
i=find(label_train==1);
img=reshape(feature_train(i(1),:),28,28);
subplot(2, 2, 2)
imshow(img)
title('1st training image in class 1')
i=find(label_test==0);
img=reshape(feature_test(i(1),:),28,28);
subplot(2, 2, 3)
imshow(img)
title('1st test image in class 0')
i=find(label_test==1);
img=reshape(feature_test(i(1),:),28,28);
subplot(2, 2, 4)
imshow(img)
title('1st test image in class 1')
