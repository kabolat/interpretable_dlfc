clc; clear all; close all;

%%
SLIDE = figure;
load celebA_trained_models_7.mat auimds_tr numTrainImages encoderNet decoder1Net decoder2Net latentDim

miniBatchSize = 10000;
auimds_tr.MiniBatchSize = miniBatchSize;
Z = []; ZMean = []; ZLogVar = [];

for ii = 1: numTrainImages/miniBatchSize
    XTrain = single(reshape(cell2mat(table2array(read(auimds_tr))),64,miniBatchSize,64,3));
    XTrain = dlarray(XTrain,'SBSC');
    [zSampled, zMean, zLogvar] = sampling(encoderNet, XTrain);
    Z = [Z extractdata(squeeze(zSampled))]; 
    ZMean = [ZMean extractdata(squeeze(zMean))]; 
    ZLogVar = [ZLogVar extractdata(squeeze(zLogvar))];
end

KL = KL_Loss(ZMean',exp(ZLogVar/2)');

%%
numRules = 5;
fcmOpts = [1.11, 1000]; %%fcmOpts(1):m %%fcmOpts(2):Max Iterations

[C,U] = fcm(Z',numRules,fcmOpts);
mfparams = permute(visualizeFCM(Z',C,U,0),[2 1 3]);

S_learned = squeeze(mfparams(:,:,1));
C_learned = squeeze(mfparams(:,:,2));


%%
[sorted_KL,indx] = sort(KL,'descend');

C_learned_sorted = C_learned(indx,:);
S_learned_sorted = S_learned(indx,:);

[~,sigmIndex_Left] = min(C_learned_sorted'); %%The leftmost center indices
[~,sigmIndex_Right] = max(C_learned_sorted'); %%The rightmost center indices
sigmIndex = [sigmIndex_Left; sigmIndex_Right];

dimNo = 7;

figure
x = linspace(-1.5,1.5,1000);
for ii = 1:dimNo
    subplot(dimNo,1,ii)
    hold on
    for jj = 1:numRules
        
        if jj == sigmIndex(1,ii)
            [a, c_s] = gauss2sigm(C_learned_sorted(ii,sigmIndex(1,ii)),S_learned_sorted(ii,sigmIndex(1,ii)),1);
            plot(x,sigmf(x,[a,c_s]),'LineWidth', 3)
        elseif jj == sigmIndex(2,ii)
            [a, c_s] = gauss2sigm(C_learned_sorted(ii,sigmIndex(2,ii)),S_learned_sorted(ii,sigmIndex(2,ii)),2);
            plot(x,sigmf(x,[a,c_s]),'LineWidth', 3)
        else
            plot(x, gaussmf(x,[S_learned_sorted(ii,jj),C_learned_sorted(ii,jj)]),...
                'LineWidth', 3);
        end
        
        
    end
    xlabel(['$z^{(' num2str(indx(ii)) ')}$'], 'Interpreter', 'Latex', 'FontSize',15)
    ylabel(['$\bf{' num2str(sorted_KL(ii)) '}$'], 'Interpreter', 'Latex', 'FontSize',13)
end


%% Traverse
sample=10; delta=0.6;
figure(SLIDE); clf;
for kk = 1:dimNo
    subplot(dimNo,1,kk)
    zChange_rand_celeba(decoder1Net, decoder2Net, latentDim, indx(kk), sample, delta)
    xlabel(['$z^{(' num2str(indx(kk)) ')}$'], 'Interpreter', 'Latex', 'FontSize',15)
    ylabel(['$\bf{' num2str(sorted_KL(kk)) '}$'], 'Interpreter', 'Latex', 'FontSize',13)
end