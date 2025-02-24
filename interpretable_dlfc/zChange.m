function RECONS = zChange(decoderNet, latentDim, knobDim, sample, delta, centerZ, fuzzyNet, redDims, redVals,KL)

C = fuzzyNet.Layers(2,1).centers;
S = fuzzyNet.Layers(2,1).sigma;

[numInputs,numRules] = size(C);
subpGrid = [2,(sample+1)*(numInputs-1)];
recnstSz = numInputs-1;
mfSz = sample+1;

RECONS = zeros(28,28,sample+1);

%% DECODER
z = dlarray(0.7071*randn(1,1,latentDim),'SSC');
if ~isempty(redDims)
    z(1,1,redDims) = redVals;
end
z(1,1,knobDim) = centerZ;

steps = zeros(1,sample,latentDim);
steps(:,:,knobDim) = delta:delta:sample*delta;
steps = steps-mean(steps);

subplot(subpGrid(1),subpGrid(2),(floor(sample/2)*recnstSz+1:floor(sample/2+1)*recnstSz))
RECONS(:,:,sample/2+1) = squeeze(gather(extractdata(sigmoid(forward(decoderNet, z)))));
imshow(RECONS(:,:,sample/2+1))
% title([ '$$z(' num2str(knobDim) ') = ' num2str(centerZ) '$$'],'Interpreter','LaTeX' )
axis equal tight
% xlabel(['"' char(classify(fuzzyNet,z)) '"'])
z_steps = squeeze(gather(extractdata(z+steps(:,:,:))))';

for ii=1:sample+1
    
    if ii<floor(sample/2+1)
        loc = ii;
        subplot(subpGrid(1),subpGrid(2),((ii-1)*recnstSz+1:ii*recnstSz))
        RECONS(:,:,loc) = squeeze(gather(extractdata(sigmoid(forward(decoderNet, z+steps(:,loc,:))))));
        imshow(RECONS(:,:,loc))
%         xlabel(['"' char(classify(fuzzyNet,z+steps(:,loc,:))) '"'])
%         title(num2str(z_steps(knobDim,loc)),'Interpreter','LaTeX')
        axis equal tight
        
    elseif ii>floor(sample/2+1)
        loc = ii-1;
        subplot(subpGrid(1),subpGrid(2),((ii-1)*recnstSz+1:ii*recnstSz))
        RECONS(:,:,loc+1) = squeeze(gather(extractdata(sigmoid(forward(decoderNet, z+steps(:,loc,:))))));
        imshow(RECONS(:,:,loc+1))
%         xlabel(['"' char(classify(fuzzyNet,z+steps(:,loc,:))) '"'])
%         title(num2str(z_steps(knobDim,loc)),'Interpreter','LaTeX')
        axis equal tight
    end
    
end

zStr = '\quad';
tempStr = squeeze(gather(extractdata(z)));
for ii = 1:length(tempStr)
    zStr = [ zStr, num2str(tempStr(ii)), '\quad'];
end
% sgtitle([ '$$z_0=\left[' zStr '\right] $$ ' ],'Interpreter','LaTeX')

%% CLASSIFIER

x = linspace(-5,5,1000);
z = squeeze(gather(extractdata(z)));
%Sliding Feature
subplot(subpGrid(1),subpGrid(2),(subpGrid(1)-1)*subpGrid(2)+1:(subpGrid(1))*subpGrid(2))
hold on
for jj = 1:numRules
    plot(x, exp(-(x-C(knobDim,jj)).^2./(2*S(knobDim,jj)^2)),...
        'LineWidth', 3);
%     plot(z(knobDim),exp(-(z(knobDim)-C(knobDim,jj)).^2./(2*S(knobDim,jj)^2)),'kd')
end
hold off
xlim([min(steps(:,:,knobDim))+centerZ,max(steps(:,:,knobDim))+centerZ])
% xlabel(['D_{KL}=' num2str(KL(knobDim))],'FontSize',15)
title(['$$z(' num2str(knobDim) ')$$'], 'Interpreter','LaTeX', 'FontSize',15)

constFeat = 1:latentDim;
constFeat(knobDim) = [];


%Constant Features
% acc = 1;
% for ii = constFeat
%     subplot(subpGrid(1),subpGrid(2),((subpGrid(1)-1)*subpGrid(2))+((acc-1)*mfSz+1:acc*mfSz))
%     hold on
%     for jj = 1:numRules
%         plot(x, exp(-(x-C(ii,jj)).^2./(2*S(ii,jj)^2)),...
%             'LineWidth', 1);
%         plot(z(ii),exp(-(z(ii)-C(ii,jj)).^2./(2*S(ii,jj)^2)),'kd')
%     end
%     hold off
%     title(['$$\bf{z(' num2str(ii) ')}$$'], 'Interpreter','LaTeX' )
%     acc = acc+1;
% %     xlabel(['KL=' num2str(KL(ii))])
% 
% end


end