function zChange_rand_celeba(decoder1Net, decoder2Net, latentDim, knobDim, sample, delta)

z = dlarray(randn(1,1,latentDim),'SSC');
steps = zeros(1,sample,latentDim);
steps(:,:,knobDim) = delta:delta:sample*delta;
steps = steps-mean(steps);

xx = forward(decoder1Net,z);
xx = dlarray(reshape(xx,4,4,32),'SSC');
SLIDE = sigmoid(forward(decoder2Net, xx));
[rws,~] = size(SLIDE);
SLIDE = [SLIDE ones(rws,5,3)];

for ii=1:sample
    xx = forward(decoder1Net,z+steps(:,ii,:));
    xx = dlarray(reshape(xx,4,4,32),'SSC');
    SLIDEnew = sigmoid(forward(decoder2Net, xx));
    SLIDE = [SLIDE SLIDEnew];
end

SLIDE = squeeze(gather(extractdata(SLIDE)));
z_steps = squeeze(gather(extractdata(z+steps(:,:,:))))';

imshow(SLIDE)
% title(['z_0=[' num2str(squeeze(gather(extractdata(z)))') ']   z('...
%     num2str(knobDim) ') -> [' num2str(z_steps(knobDim,:)) ']'], 'FontSize', 8)
end