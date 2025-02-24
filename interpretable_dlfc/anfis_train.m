clc; clear all; close all;

%% ANFIS TRAIN
FCM = figure;
load ./Trained Networks/MNISTData.mat
load ./Trained Networks/VAEbeta001dim10_newArch.mat

XVal = XTrain(:,:,:,1:1e4);
XTrain = XTrain(:,:,:,1e4+1:end);
YVal = YTrain(1:1e4);
YTrain = YTrain(1e4+1:end);

%% Inputs
[z_fcm_train, zMean, zSigma] = visualizeLatentSpaceMeanVar(XTrain, YTrain, encoderNet, latentDim,1);
[z_fcm_val, zMeanVal, zSigmaVal] = visualizeLatentSpaceMeanVar(XVal, YVal, encoderNet, latentDim,0);
[z_fcm_test, zMeanTest, zSigmaTest] = visualizeLatentSpaceMeanVar(XTest, YTest, encoderNet, latentDim,0);

KL = KL_Loss(zMean,zSigma);

numRules = 10;
fcmOpts = [1.15, 1000]; %%fcmOpts(1):m %%fcmOpts(2):Max Iterations

figure(FCM); clf;
[C,U] = fcm(z_fcm_train,numRules,fcmOpts);
mfparams = permute(visualizeFCM(z_fcm_train,C,U,1),[2 1 3]);

%%Image
z_fcm_train = permute(z_fcm_train',[3 4 1 2]);
z_fcm_val = permute(z_fcm_val',[3 4 1 2]);
z_fcm_test = permute(z_fcm_test',[3 4 1 2]);

zMeanTrain = permute(zMean',[3 4 1 2]);
zMeanVal = permute(zMeanVal',[3 4 1 2]);
zMeanTest = permute(zMeanTest',[3 4 1 2]);

zSigmaTrain = permute(zSigma',[3 4 1 2]);
zSigmaVal = permute(zSigmaVal',[3 4 1 2]);
zSigmaTest = permute(zSigmaTest',[3 4 1 2]);

inputTrain = cat(1,zMeanTrain,zSigmaTrain);
inputVal = cat(1,zMeanVal,zSigmaVal);
inputTest = cat(1,zMeanTest,zSigmaTest);

%% Training

layers=[
    imageInputLayer([2 1 latentDim],'Name','mean_input','Normalization','none')
    nonSingletonFuzzyLayer(10,'FuzzyLayer',mfparams,"sugeno")
%     modifiedFuzzyLayerImg(10,'FuzzyLayer',mfparams,"sugeno")
%     FuzzyLayerImg(10,'FuzzyLayer',latentDim,numRules,"")
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];


maxEpochs = 200;
miniBatchSize = 1000;
options = trainingOptions('adam', ...
    'miniBatchSize',miniBatchSize, ...
    'ValidationData',{inputVal,YVal},...
    'ValidationFrequency',25,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate', 1e-2,...
    'MaxEpochs',maxEpochs, ...
    'Plots','training-progress',...
    'L2Regularization',1e-6);

fuzzyNet = trainNetwork(inputTrain,YTrain,layers,options);

%% Confusion Chart
YPred = classify(fuzzyNet,inputTest);
accuracy = sum(YPred==YTest)/length(YPred);
figure
cm = confusionchart(YTest,YPred, ...
    'Title','Confusion Chart', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% LEARNED Centers and Sigmas

[sorted_KL,indx] = sort(KL,'descend');

C_learned = fuzzyNet.Layers(2,1).centers;
S_learned = fuzzyNet.Layers(2,1).sigma;

C_learned = C_learned(indx,:);
S_learned = S_learned(indx,:);

figure
x = linspace(-3,3,1000);
for ii = 1:6
    subplot(6,1,ii)
    hold on
    for jj = 1:numRules
        plot(x, gaussmf(x,[S_learned(ii,jj),C_learned(ii,jj)]),...
            'LineWidth', 3);
    end
    hold off
    xlabel(['$z^{(' num2str(indx(ii)) ')}$'], 'Interpreter', 'Latex', 'FontSize',15)
    ylabel(['$\bf{KL=' num2str(sorted_KL(ii)) '}$'], 'Interpreter', 'Latex', 'FontSize',13)
end

%%
SLIDE1 = figure;
SLIDE2 = figure;
%% Sliding on One z Dimension
knobDim=2; sample=10; delta=0.6;
figure(SLIDE1); clf;
zChange(decoderNet, latentDim, knobDim, sample, delta, 0, fuzzyNet, [1 3:10], -3*ones(1,9), KL);
% figure(SLIDE2); clf;
% z_change_all(decoderNet, latentDim, sample, delta)

%%
MISC = figure;
RULES = figure;
%% Show A Random Misclassification
IDX = 1:length(YTest);
mis_idx = IDX(~(YTest==YPred));
rand_mis_idx = mis_idx(randi(length(mis_idx)));
rand_mis_z = z_fcm_test(:,:,:,rand_mis_idx);

figure(MISC);clf;
subplot(1,2,1)
imshow(squeeze(gather(extractdata(XTest(:,:,:,rand_mis_idx)))))
subplot(1,2,2)
imshow(squeeze(gather(extractdata(sigmoid(forward(decoderNet, dlarray(rand_mis_z,'SSC')))))))
sgtitle(['y_{desired}:"' char(YTest(rand_mis_idx)) '" y_{pred}:"'...
    char(YPred(rand_mis_idx)) '"'])

figure(RULES);clf;
x = linspace(-2,2,1000);
acc = 1; rule_fire = 1;
rand_mis_z = squeeze(rand_mis_z);
for rule=1:numRules
    for dim=1:latentDim
        mu = exp(-(rand_mis_z(dim)-C_learned(dim,rule)).^2./(2*S_learned(dim,rule)^2));
        subplot(numRules,latentDim+1,acc)
        hold on
        plot(x,exp(-(x-C_learned(dim,rule)).^2./(2*S_learned(dim,rule)^2)),'LineWidth',2)
        plot(rand_mis_z(dim),mu,'kd')
        hold off
        rule_fire = mu*rule_fire;
        acc = acc+1;
        if rule == 1
            title(num2str(KL(dim)))
        end
    end
    
    subplot(numRules,latentDim+1,acc)
    bar(rule_fire)
    ylim([0 0.1])
    title(['f_' num2str(rule) ':' num2str(rule_fire)])
    acc = acc+1; rule_fire=1;
end

%% Slide From Misclassification
knobDim=4; sample=10; delta=.4;
figure(SLIDE1); clf;
zChange(decoderNet, latentDim, knobDim, sample, delta, rand_mis_z(knobDim), fuzzyNet, 1:latentDim, rand_mis_z,KL)


