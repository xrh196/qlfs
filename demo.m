clc
clear
close all
load vehicle
D=zscore(data);
new_data=[D label];
z=size(D,2);
n=18;
[M,N]=size(new_data);
indices=crossvalind('Kfold',new_data(1:M,N),5);
for iter=1:5
test=(indices==iter);
train =~test;
TrainingData=new_data(train,:); 
TestingData=new_data(test,:);
X=[(TrainingData(:,1:end-1))];
Y=TrainingData(:,end);
[r]=QLFS(TrainingData);r=r';
i1=0;
for i=1:1:n
i1=i1+1;
tzno=i;     
TrainingData1=[TrainingData(:,r(1:tzno)),TrainingData(:,end)];
TestingData1=[TestingData(:,r(1:tzno)),TestingData(:,end)]; 
[accuracy(i1,iter),~]=f_SVM(TrainingData1,TestingData1);                                                     
end
end
accuracy=mean(accuracy,2);
plot(accuracy(1:1:i1),'-*');
legend('QLFS');
hold on;
axis([1 i1 0.5 1])
ylabel('AC') 
xlabel('number of features')  ;
set(gca, 'Fontname', 'Times newman', 'Fontsize', 18);