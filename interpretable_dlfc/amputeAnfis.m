clc; clear all ; close all;

LIB = {'anfisbeta001dim10_newArch_clst3_mean.mat';
    'anfisbeta001dim10_newArch_clst3.mat';
    'anfisbeta001dim10_newArch_clst5_mean.mat'
    'anfisbeta001dim10_newArch_clst5.mat';
    'anfisbeta001dim10_newArch_clst10_mean.mat';
    'anfisbeta001dim10_newArch_clst10.mat'};

%% AMPUTE TRAIN

load ACC

for kk = 1
load(LIB{kk})
load MNISTData.mat

XVal = XTrain(:,:,:,1:1e4);
XTrain = XTrain(:,:,:,1e4+1:end);
YVal = YTrain(1:1e4);
YTrain = YTrain(1e4+1:end);

%% Data Preperation

% [z_fcm_train, zMean, zVar] = visualizeLatentSpaceMeanVar(XTrain, YTrain, encoderNet, latentDim,0);
% [z_fcm_val,zMeanVal] = visualizeLatentSpaceMeanVar(XVal, YVal, encoderNet, latentDim,0);
% [z_fcm_test,zMeanTest] = visualizeLatentSpaceMeanVar(XTest, YTest, encoderNet, latentDim,0);

% KL = KL_Loss(zMean,zVar);

%%Amputation
[sorted_KL, indx_KL] = sort(KL,'descend');
cum_KL = cumsum(sorted_KL);
sum_KL = cum_KL(end);
cum_KL = cum_KL/sum_KL;

for jj = 3
thr = jj;

ampt_z_indx = indx_KL(1:thr);

numInputs = length(ampt_z_indx);

%Data prep. for training with mean
% z_fcm_train = zMean;
% z_fcm_val = zMeanVal;
% z_fcm_test = zMeanTest;

z_fcm_trainn = z_fcm_train(:,:,ampt_z_indx,:);
z_fcm_vall = z_fcm_val(:,:,ampt_z_indx,:);
z_fcm_testt = z_fcm_test(:,:,ampt_z_indx,:);

%%MF Reduction
mfparamss = mfparams(ampt_z_indx,:,:);


%% Training

layers=[
    imageInputLayer([1 1 numInputs],'Name','input_anfis','Normalization','none')
    modifiedFuzzyLayerImg(10,'FuzzyLayer',mfparamss,"sugeno")
    softmaxLayer
    classificationLayer];

maxEpochs = 25;
miniBatchSize = 1000;
options = trainingOptions('adam', ...
    'miniBatchSize',miniBatchSize, ...
    'ValidationData',{z_fcm_vall,YVal},...
    'ValidationFrequency',25,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu', ...
    'InitialLearnRate', 1e-2,...
    'MaxEpochs',maxEpochs, ...
    'Plots','training-progress',...
    'L2Regularization',1e-6);

amputeNet = trainNetwork(z_fcm_trainn,YTrain,layers,options);

%% Confusion Chart
YPred = classify(amputeNet,z_fcm_testt);
accuracy = sum(YPred==YTest)/length(YPred)

ACC(jj,kk) = accuracy;
end
end

% figure
% cm = confusionchart(YTest,YPred, ...
%     'Title','Confusion Chart', ...
%     'RowSummary','row-normalized', ...
%     'ColumnSummary','column-normalized');

%% LEARNED Centers and Sigmas
% 
% C_learned = amputeNet.Layers(2,1).centers;
% S_learned = amputeNet.Layers(2,1).sigma;
% 
% figure
% x = linspace(-5,5,1000);
% for ii = 1:numInputs
%     subplot(numInputs,1,ii)
%     hold on
%     for jj = 1:numRules
%         plot(x, exp(-(x-C_learned(ii,jj)).^2./(2*S_learned(ii,jj)^2)),...
%             'LineWidth', 3);
%     end
%     hold off
%     ylabel(num2str(KL(ampt_z_indx(ii))))
% end
% 







