clc; close all; clear all;

%%
numTrainImages = 18e4;

imds = imageDatastore('./img_align_celeba');
train_imds = subset(imds,1:numTrainImages);
val_imds = subset(imds,numTrainImages+1:numpartitions(imds));

numValImages = numpartitions(val_imds);

auimds_tr = augmentedImageDatastore([64 64],train_imds);
auimds_val = augmentedImageDatastore([64 64],val_imds);

XTrain = single(reshape(cell2mat(table2array(readall(auimds_tr))),64,numTrainImages,64,3));
mean_tr = squeeze(mean(XTrain,2));
std_tr = squeeze(std(XTrain,1,2));
clear XTrain

%%
latentDim = 15;
Beta = 120;
imageSize = [64 64 3];

encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','zscore','Mean',mean_tr,'StandardDeviation',std_tr)
    convolution2dLayer(4, 32, 'Padding',1, 'Stride', 2, 'Name', 'conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(4, 32, 'Padding',1, 'Stride', 2, 'Name', 'conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(4, 32, 'Padding',1, 'Stride', 2, 'Name', 'conv3')
    reluLayer('Name','relu3')
    convolution2dLayer(4, 32, 'Padding',1, 'Stride', 2, 'Name', 'conv4')
    reluLayer('Name','relu4')
    fullyConnectedLayer(256, 'Name', 'fc_encoder1')
    reluLayer('Name','relu5')
    fullyConnectedLayer(256, 'Name', 'fc_encoder2')
    reluLayer('Name','relu6')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder3')
    ]);

decoder1LG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    fullyConnectedLayer(256, 'Name', 'fc_decoder1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(256, 'Name', 'fc_decoder2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(512, 'Name', 'fc_decoder3')
    reluLayer('Name','relu3')
    ]);
decoder2LG = layerGraph([
    imageInputLayer([4, 4, 32],'Name','i','Normalization','none')
    transposedConv2dLayer(4, 32, 'Cropping', 1, 'Stride', 2, 'Name', 'transpose1')
    reluLayer('Name','relu3')
    transposedConv2dLayer(4, 32, 'Cropping', 1, 'Stride', 2, 'Name', 'transpose2')
    reluLayer('Name','relu4')
    transposedConv2dLayer(4, 32, 'Cropping', 1, 'Stride', 2, 'Name', 'transpose3')
    reluLayer('Name','relu5')
    transposedConv2dLayer(4, 3, 'Cropping', 1, 'Stride', 2, 'Name', 'transpose5')
    ]);

encoderNet = dlnetwork(encoderLG);
decoder1Net = dlnetwork(decoder1LG);
decoder2Net = dlnetwork(decoder2LG);

%%
lossGraph = figure;
reconst = figure;
randLatent = figure;
FCM = figure;
SLIDE = figure;
crossLossGraph =figure;

%%
numEpochs = 200;
miniBatchSize = 128;
lr = 1e-4;      %Learning Rate
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder1 = [];
avgGradientsSquaredDecoder1 = [];
avgGradientsDecoder2 = [];
avgGradientsSquaredDecoder2 = [];

itx = 1:numEpochs*numIterations;
elbo_arr = 0*itx; rLoss_arr = 0*itx; KL_arr = 0*itx;
elbo_arr_val = zeros(1,numEpochs); rLoss_arr_val = elbo_arr_val; KL_arr_val = elbo_arr_val;
acc = 1;

%%
for epoch = 101:numEpochs
    tic;
    auimds_tr = shuffle(auimds_tr);
    auimds_val = shuffle(auimds_val);
        
    XVal = reshape(cell2mat(table2array(read(auimds_val))),64,128,64,3);
    XVal = dlarray(single(XVal), 'SBSC');
    XVal = gpuArray(XVal);
    
    for i = 1:numIterations
        iteration = iteration + 1;
        
        XBatch = reshape(cell2mat(table2array(read(auimds_tr))),64,128,64,3);
        XBatch = dlarray(single(XBatch), 'SBSC');
        XBatch = gpuArray(XBatch);
        
        %%Gradyan Hesabi
        [infGrad, genGrad1, genGrad2, z, zMean, zLogvar, xPred, elbo, rLoss, KLavg] = dlfeval(...
            @modelGradients, encoderNet, decoder1Net, decoder2Net, XBatch, Beta);
        
        %%Adam
        [decoder2Net.Learnables, avgGradientsDecoder2, avgGradientsSquaredDecoder2] = ...
            adamupdate(decoder2Net.Learnables, ...
            genGrad2, avgGradientsDecoder2, avgGradientsSquaredDecoder2, iteration, lr);
        [decoder1Net.Learnables, avgGradientsDecoder1, avgGradientsSquaredDecoder1] = ...
            adamupdate(decoder1Net.Learnables, ...
            genGrad1, avgGradientsDecoder1, avgGradientsSquaredDecoder1, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
            infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
        
        elbo_arr(acc) = squeeze(gather(extractdata(elbo)));
        rLoss_arr(acc)= squeeze(gather(extractdata(rLoss)));
        KL_arr(acc)   = squeeze(gather(extractdata(KLavg)));
        
        if mod(acc,100) == 0
                disp("Epoch : "+epoch+" Mean ELBO loss = "+gather(elbo_arr(acc)))
        end
        
        acc = acc+1;
        
    end
    elapsedTime = toc;
    
    
    %%Validation
    [~, ~, ~, ~, ~, ~, ~, elboVal, rLossVal, KLVal] = dlfeval(...
        @modelGradients, encoderNet, decoder1Net, decoder2Net, XVal, Beta);
    elbo_arr_val(epoch)=squeeze(gather(extractdata(elboVal)));
    rLoss_arr_val(epoch)=squeeze(gather(extractdata(rLossVal)));
    KL_arr_val(epoch)=squeeze(gather(extractdata(KLVal)));
    
    disp("Epoch : "+epoch+" Mean ELBO loss = "+gather(mean(elbo_arr(acc-numIterations:acc-1)))+...
        ". Time taken for epoch = "+ elapsedTime + "s")
    
    figure(lossGraph)
    subplot 311
    hold on
    plot(numIterations*(1:numEpochs),elbo_arr_val,'r*','LineWidth',1)
    hold off
    subplot 312
    hold on
    plot(numIterations*(1:numEpochs),rLoss_arr_val,'b*','LineWidth',1)
    hold off
    subplot 313
    hold on
    plot(numIterations*(1:numEpochs),KL_arr_val,'g*','LineWidth',1)
    hold off
    
    
    %%Epoch Visualization
    
    %%Traverse
    sample=10; delta=0.4;
    figure(SLIDE); clf;
    for kk = 1:latentDim
        subplot(latentDim,1,kk)
        zChange_rand_celeba(decoder1Net, decoder2Net, latentDim, kk, sample, delta)
    end
    
    disp(KL_Loss(permute(extractdata(zMean),[2 1]),permute(exp(extractdata(zLogvar)/2),[2 1])))
    
end

save celebA_trained_models_7

%%
% numClst = 5;
% [z_fcm, z_fcm_mean, z_fcm_var] = visualizeLatentSpaceMeanVar(XTrain, YTrain, encoderNet, latentDim,1);
% fcmOpts = [numClst, 1.2, 1000]; %%fcmOpts(1):Number of Clusters %%fcmOpts(2):m %%fcmOpts(3):Max Iterations
% figure(FCM); clf;
% [C,U] = latent_fcm(z_fcm,fcmOpts);
% mfparams = visualizeFCM(z_fcm,C,U);

%%
% knobDim=1; sample=20; delta=0.2;
% figure(SLIDE); clf;
% zChange_rand(decoderNet, latentDim, knobDim, sample, delta)
