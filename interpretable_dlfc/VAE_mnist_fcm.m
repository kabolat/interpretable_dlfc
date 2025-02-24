clc; close all; clear all;

%%
trainImagesFile = '/home/kutaybolat/Documents/MATLAB/DATA/MNIST/train-images.idx3-ubyte';
trainLabelsFile = '/home/kutaybolat/Documents/MATLAB/DATA/MNIST/train-labels.idx1-ubyte';
testImagesFile = '/home/kutaybolat/Documents/MATLAB/DATA/MNIST/t10k-images.idx3-ubyte';
testLabelsFile = '/home/kutaybolat/Documents/MATLAB/DATA/MNIST/t10k-labels.idx1-ubyte';

numTrainImages = 6e4;

XTrain = processMNISTimages(trainImagesFile);
XTrain = XTrain(:,:,:,1:numTrainImages);
XTrain = dlarray(single(XTrain), 'SSCB');
XTrain = gpuArray(XTrain);
YTrain = processMNISTlabels(trainLabelsFile);
YTrain = YTrain(1:numTrainImages);
YTrain_d = double(string(YTrain));

XTest = processMNISTimages(testImagesFile);
XTest = XTest(:,:,:,1:floor(numTrainImages/6));
YTest = processMNISTlabels(testLabelsFile);
YTest = YTest(1:floor(numTrainImages/6));
YTest_d = double(string(YTest));

%%
latentDim = 10;
Beta = 0.01;
imageSize = [28 28 1];

encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer(7, 16, 'Padding','same', 'Stride', 2, 'Name', 'conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(5, 32, 'Padding','same', 'Stride', 1, 'Name', 'conv3')
    reluLayer('Name','relu3')
    convolution2dLayer(3, 64, 'Padding','same', 'Stride', 2, 'Name', 'conv2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
    ]);

decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    transposedConv2dLayer(7, 64, 'Cropping', 'same', 'Stride', 7, 'Name', 'transpose1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(5, 32, 'Cropping', 'same', 'Stride', 1, 'Name', 'transpose2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(3, 16, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(3, 1, 'Cropping', 'same', 'Stride', 2, 'Name', 'transpose4')
    ]);

encoderNet = dlnetwork(encoderLG);
decoderNet = dlnetwork(decoderLG);

%%
lossGraph = figure;
reconst = figure;
randLatent = figure;
FCM = figure;
SLIDE = figure;
crossLossGraph =figure;

%%
numEpochs = 200;
miniBatchSize = 1200;
lr = 1e-3;      %Learning Rate
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];

XPredEpch = zeros(28,28*10,numEpochs);

itx = 1:numEpochs*numIterations;
elbo_arr = 0*itx; rLoss_arr = 0*itx; KL_arr = 0*itx;
elbo_arr_val = zeros(1,numEpochs); rLoss_arr_val = elbo_arr_val; KL_arr_val = elbo_arr_val;
acc = 1;

%%
for epoch = 1:numEpochs
    tic;
    shfl = randperm(length(YTrain_d));
    XTrainShuffle = XTrain(:,:,:,shfl);
    YTrainShuffle = YTrain_d(shfl);
    
    for i = 1:numIterations
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize; %%Datanin basindan sirayla aliyor
        XBatch = XTrainShuffle(:,:,:,idx);
        YBatch = YTrainShuffle(idx);
%         XBatch = dlarray(single(XBatch), 'SSCB');
        
%         XBatch = gpuArray(XBatch);
        
        %%Gradyan Hesabi
        [infGrad, genGrad, z, zMean, zLogvar, xPred, elbo, rLoss, KLavg] = dlfeval(...
            @modelGradients, encoderNet, decoderNet, XBatch, Beta);
        
        %%Adam
        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
            genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
            infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
        
%         %%Iteration Visualization
%         elbo_arr(acc) = squeeze(gather(extractdata(elbo)));
%         rLoss_arr(acc)= squeeze(gather(extractdata(rLoss)));
%         KL_arr(acc)   = squeeze(gather(extractdata(KLavg)));
%         
%         figure(lossGraph)
%         subplot 311
%         hold on
%         plot(itx,elbo_arr,'k.','LineWidth',0.5)
%         ylim([0 2*median(elbo_arr(1:acc))])
%         xlim([0 acc])
%         hold off
%         subplot 312
%         hold on
%         plot(itx,rLoss_arr,'k.','LineWidth',0.5)
%         ylim([0 2*median(rLoss_arr(1:acc))])
%         xlim([0 acc])
%         hold off
%         subplot 313
%         hold on
%         plot(itx,KL_arr,'k.','LineWidth',0.5)
%         ylim([0 2*median(KL_arr(1:acc))])
%         xlim([0 acc])
%         hold off
%
acc = acc+1;

    end
    elapsedTime = toc;
    
    
    %%Validation
    [~, ~, ~, ~, ~, ~, elboVal, rLossVal, KLVal] = dlfeval(...
        @modelGradients, encoderNet, decoderNet, XTest, Beta);
    elbo_arr_val(epoch)=squeeze(gather(extractdata(elboVal)));
    rLoss_arr_val(epoch)=squeeze(gather(extractdata(rLossVal)));
    KL_arr_val(epoch)=squeeze(gather(extractdata(KLVal)));
    
    disp("Epoch : "+epoch+" Mean ELBO loss = "+gather(mean(elbo_arr(acc-numIterations:acc-1)))+...
        ". Time taken for epoch = "+ elapsedTime + "s")
    
    figure(lossGraph)
    subplot 311
    hold on
    plot(numIterations*(1:numEpochs),elbo_arr_val,'ro','LineWidth',1)
    hold off
    subplot 312
    hold on
    plot(numIterations*(1:numEpochs),rLoss_arr_val,'ro','LineWidth',1)
    hold off
    subplot 313
    hold on
    plot(numIterations*(1:numEpochs),KL_arr_val,'ro','LineWidth',1)
    hold off
    
    
    %%Epoch Visualization
    %%Reconstruction
    figure(reconst)
    title("Reconstructed Images for Epoch "+epoch)
    XPredEpch(:,:,epoch) = visualizeReconstruction(XTrain, YTrain, encoderNet, decoderNet);
    drawnow
    
    %%Sampled Latent Space
    figure(randLatent)
    title("Randomly Sampled Latent Space for Epoch "+epoch)
    z_fcm = visualizeLatentSpace(XTrain, YTrain, encoderNet);
    drawnow
    
    %%Traverse
    knobDim=1; sample=20; delta=0.2;
    figure(SLIDE); clf;
    zChange_rand(decoderNet, latentDim, knobDim, sample, delta)
    
end

%%
numClst = 5;
[z_fcm, z_fcm_mean, z_fcm_var] = visualizeLatentSpaceMeanVar(XTrain, YTrain, encoderNet, latentDim,1);
fcmOpts = [numClst, 1.2, 1000]; %%fcmOpts(1):Number of Clusters %%fcmOpts(2):m %%fcmOpts(3):Max Iterations
figure(FCM); clf;
[C,U] = latent_fcm(z_fcm,fcmOpts);
mfparams = visualizeFCM(z_fcm,C,U);

%%
knobDim=1; sample=20; delta=0.2;
figure(SLIDE); clf;
zChange_rand(decoderNet, latentDim, knobDim, sample, delta)



