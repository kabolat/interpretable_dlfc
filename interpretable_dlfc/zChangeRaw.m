function RECONS = zChangeRaw(decoderNet, latentDim, knobDim, sample, delta, centerZ, redDims,redVals)

RECONS = zeros(28,28,sample+1);

%% DECODER
z = dlarray(randn(1,1,latentDim),'SSC');
if ~isempty(redDims)
    z(1,1,redDims) = redVals;
end
z(1,1,knobDim) = centerZ;

steps = zeros(1,sample,latentDim);
steps(:,:,knobDim) = delta:delta:sample*delta;
steps = steps-mean(steps);

RECONS(:,:,sample/2+1) = squeeze(gather(extractdata(sigmoid(forward(decoderNet, z)))));

for ii=1:sample+1
    
    if ii<floor(sample/2+1)
        loc = ii;
        RECONS(:,:,loc) = squeeze(gather(extractdata(sigmoid(forward(decoderNet, z+steps(:,loc,:))))));
    elseif ii>floor(sample/2+1)
        loc = ii-1;
        RECONS(:,:,loc+1) = squeeze(gather(extractdata(sigmoid(forward(decoderNet, z+steps(:,loc,:))))));
    end
    
end

end