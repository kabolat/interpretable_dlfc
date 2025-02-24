function z_change_all(decoderNet, latentDim, sample, delta)

z = dlarray(sqrt(1/2)*randn(1,1,latentDim),'SSC');

acc = 1;
for kk = 1:latentDim
    steps = zeros(1,sample,latentDim);
    steps(:,:,kk) = delta:delta:sample*delta;
    steps = steps-mean(steps);
    
    for ii=1:sample+1
        
        if ii<floor(sample/2+1)
            loc = ii;
            subplot(latentDim,sample+1,acc)
            imshow(squeeze(gather(extractdata(sigmoid(forward(decoderNet, z+steps(:,loc,:)))))))
            axis equal tight
            
        elseif ii==floor(sample/2+1)
            subplot(latentDim,sample+1,acc)
            imshow(squeeze(gather(extractdata(sigmoid(forward(decoderNet, z))))))
            axis equal tight
            
        elseif ii>floor(sample/2+1)
            loc = ii-1;
            subplot(latentDim,sample+1,acc)
            imshow(squeeze(gather(extractdata(sigmoid(forward(decoderNet, z+steps(:,loc,:)))))))
            axis equal tight
        end
        
        acc = acc+1;
        
    end
end
end