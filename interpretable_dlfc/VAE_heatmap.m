clc; clear all;

%%
load anfisbeta001dim10_newArch_clst3.mat
close all;

%%
numOfSamples = 10000;
meanDiff = zeros(28,28,numOfSamples);
MEANDiff = zeros(28,28,latentDim);

%%
for ii = 1:latentDim
    %%
    for jj = 1:numOfSamples
        %%
        SLIDE = figure;
        %%
        knobDim=ii; sample=10; delta=0.4;
        figure(SLIDE); clf;
        RECONS = zChange(decoderNet, latentDim, knobDim, ...
            sample, delta, 0, fuzzyNet, [], [], KL);
%         RECONS = zChangeRaw(decoderNet,latentDim,knobDim,sample,delta,0,[],[]);
        
        %%
        SLIDE_diff = figure;
        %%
        DIFF = diff(RECONS,1,3)/delta;
        cmin=min(min(min(DIFF))); cmax=max(max(max(DIFF)));
        
        figure(SLIDE_diff);clf;
        for kk = 1:sample
            subplot(1,sample,kk)
            imagesc(DIFF(:,:,kk),[cmin cmax])
            axis equal tight off
            colormap hot
        end
        
        %%
        MEAN_diff = figure;
        %%
        meanDiff(:,:,jj) = mean(DIFF,3);
        
        figure(MEAN_diff); clf;
        
        imagesc(meanDiff(:,:,jj))
        colorbar
        
    end
    
    MEANDiff(:,:,ii) = mean(meanDiff,3);
    
    imagesc(MEANDiff(:,:,ii))
    colorbar
    drawnow
end

% C = reshape(MEANDiff,size(MEANDiff,2),[],1);
% imagesc(C)
% axis equal tight off

%%
MEANDiff = reshape(C,28,28,10);
% cmin=min(min(min(C))); cmax=max(max(max(C)));
for ii = 1:latentDim
    subplot(1,latentDim,ii)
    imagesc(MEANDiff(:,:,ii))
    axis equal tight off
%     title(['\Deltaz(' num2str(ii) ') KL=' num2str(KL(ii))])
    colormap hot
    colorbar
end

