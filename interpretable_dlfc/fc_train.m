clc; clear all; close all;

%% FULLY CONNECTED TRAIN

load MNISTData.mat
load VAEbeta001dim10_newArch.mat

XVal = XTrain(:,:,:,1:1e4);
XTrain = XTrain(:,:,:,1e4+1:end);
YVal = YTrain(1:1e4);
YTrain = YTrain(1e4+1:end);

%%
[z_fcm_train, zMean, zVar] = visualizeLatentSpaceMeanVar(XTrain, YTrain, encoderNet, latentDim,0);
[z_fcm_val,zMeanVal] = visualizeLatentSpaceMeanVar(XVal, YVal, encoderNet, latentDim,0);
[z_fcm_test,zMeanTest] = visualizeLatentSpaceMeanVar(XTest, YTest, encoderNet, latentDim,0);

KL = KL_Loss(zMean,zVar);

z_fcm_train = zMean;
z_fcm_val = zMeanVal;
z_fcm_test = zMeanTest;

z_fcm_train = permute(z_fcm_train',[3 4 1 2]);
z_fcm_val = permute(z_fcm_val',[3 4 1 2]);
z_fcm_test = permute(z_fcm_test',[3 4 1 2]);

%%
layers=[
    imageInputLayer([1 1 latentDim],'Name','input_anfis','Normalization','none')
    fullyConnectedLayer(51)
    tanhLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

maxEpochs = 25;
miniBatchSize = 1000;
options = trainingOptions('adam', ...
    'miniBatchSize',miniBatchSize, ...
    'ValidationData',{z_fcm_val,YVal},...
    'ValidationFrequency',25,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu', ...
    'InitialLearnRate', 1e-2,...
    'MaxEpochs',maxEpochs, ...
    'Plots','training-progress',...
    'L2Regularization',1e-6);

fullyNet = trainNetwork(z_fcm_train,YTrain,layers,options);

%% Confusion Chart
YPred = classify(fullyNet,z_fcm_test);
accuracy = sum(YPred==YTest)/length(YPred);
figure
cm = confusionchart(YTest,YPred, ...
    'Title','Confusion Chart', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
