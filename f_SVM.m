function [A,TrA]=f_SVM(tr,te)

%Input:      tr: Training set
%            te: testing set
%            Note that: each row represents a instance, last column is label, begins from 1
%Output:     A: Testing Accuracy
%            TrA: Training Accuracy  
%            svmlabel: predict label by svm for testingdata

Trd = tr(:,1:end-1);
Trl = tr(:,end);
Ted = te(:,1:end-1);
Tel = te(:,end);

% %% 归一化预处理
% [Trd1,Ted1] = scaleForSVM(trd1,ted1,0,1);

% [bestCVaccuracy,bestc,bestg] = SVMcgForClass(Trl,Trd)

% ga_option.maxgen = 100;
% ga_option.sizepop = 20; 
% ga_option.ggap = 0.9;
% ga_option.cbound = [0,100];
% ga_option.gbound = [0,100];
% ga_option.v = 5;
% [bestacc,bestc,bestg] = gaSVMcgForClass(train_data_labels,train_final,ga_option)

% pso_option.c1 = 1.5;
% pso_option.c2 = 1.7;
% pso_option.maxgen = 100;
% pso_option.sizepop = 20;
% pso_option.k = 0.6;
% pso_option.wV = 1;
% pso_option.wP = 1;
% pso_option.v = 3;
% pso_option.popcmax = 100;
% pso_option.popcmin = 0.1;
% pso_option.popgmax = 100;
% pso_option.popgmin = 0.1;
% [bestacc,bestc,bestg] = psoSVMcgForClass(Trl1,Trd1,pso_option)
% 

% cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),'-t', num2str(2)];
% cmd = ['-c ',num2str(16),' -g ',num2str(0.0625)];
% cmd = ['-c ',num2str(5.278),' -g ',num2str(0.1895)];


% cmd = ['-c ',num2str(16),' -g ',num2str(0.0068)];

%% 分类预测,cmd
model = svmtrain(Trl, Trd, '-t 0'); %#ok<SVMTRAIN>
[ptrain_label, TrainingAccuracy,~] = svmpredict(Trl, Trd, model);
[svmlabel, TestingAccuracy,~] = svmpredict(Tel, Ted, model);

TrA=TrainingAccuracy(1)/100;
A=TestingAccuracy(1)/100;


end

