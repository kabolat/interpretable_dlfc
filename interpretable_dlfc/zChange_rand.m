function zChange_rand(decoderNet, latentDim, knobDim, sample, delta)

z = dlarray(randn(1,1,latentDim),'SSC');
steps = zeros(1,sample,latentDim);
steps(:,:,knobDim) = delta:delta:sample*delta;
steps = steps-mean(steps);


SLIDE = sigmoid(forward(decoderNet, z));
[rws,~] = size(SLIDE);
SLIDE = [SLIDE ones(rws,5)];

for ii=1:sample
    SLIDE = [SLIDE sigmoid(forward(decoderNet, z+steps(:,ii,:)))];
end

SLIDE = squeeze(gather(extractdata(SLIDE)));
z_steps = squeeze(gather(extractdata(z+steps(:,:,:))))';

imshow(SLIDE)
title(['z_0=[' num2str(squeeze(gather(extractdata(z)))') ']   z('...
    num2str(knobDim) ') -> [' num2str(z_steps(knobDim,:)) ']'], 'FontSize', 8)
end