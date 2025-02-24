function XPred09 = visualizeReconstruction(XTest,YTest,encoderNet,decoderNet)

XPred09 = zeros(size(XTest,1),10*size(XTest,2));
    for c=0:9
        idx = iRandomIdxOfClass(YTest,c);
        X = XTest(:,:,:,idx);
        
        [z, ~, ~] = sampling(encoderNet, X);
        XPred = sigmoid(forward(decoderNet, z));
        
        XPred = squeeze(gather(extractdata(XPred)));
        XPred09(:,size(XPred,2)*c+1:size(XPred,2)*(c+1)) =  XPred;
    end
    imshow(XPred09)
end

function idx = iRandomIdxOfClass(T,c)
idx = T == categorical(c);
idx = find(idx);
idx = idx(randi(numel(idx),1));
end