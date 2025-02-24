classdef nonSingletonFuzzyLayer < nnet.layer.Layer
    
    properties (Learnable)
        b
    end
    
    properties
        centers
        sigma
        rules
        inputs
        type
        outputs
    end
    
    methods
        function layer = nonSingletonFuzzyLayer(N,name,mfParams,Type)
                        
            layer.Name = name;
            % Initialize layer weights.
            layer.centers = mfParams(:,:,2);
            layer.sigma  = mfParams(:,:,1);
            
            [layer.inputs,layer.rules,~] = size(mfParams);
            layer.type = Type;
            layer.outputs = N;
            
            if Type == "sugeno"
                layer.b = randn(N,layer.inputs+1,layer.rules);
            else
                layer.b = randn(N,layer.rules);
            end
        end
        
        
        function Y = predict(layer,X)
            
            x_M = X(1,:,:,:);
            x_S = X(2,:,:,:);
            
            [h,w,c,n] = size(x_M);
            C = permute(repmat(layer.centers,1,1,h,w,n),[3 4 1 5 2]);
            S = permute(repmat(layer.sigma,1,1,h,w,n),[3 4 1 5 2]);
            
            MU = exp( -(C-repmat(x_M,1,1,1,1,layer.rules)).^2 ./ (2* (S.^2+(repmat(x_S,1,1,1,1,layer.rules).^2) ) ) );
            
            %% Calculate The Firing Levels
            
            F = prod(MU,3); F_norm = F./sum(F,5); %%NORMALIZED FIRING LEVEL VECTOR
            
            %% Calculate The Output
            if layer.type == "sugeno"
                X_sug = permute(repmat(cat(3,x_M,ones(h,w,1,n)),1,1,1,1,layer.outputs,layer.rules),[1 2 5 3 4 6]);
                B = permute(repmat(layer.b,1,1,1,h,w,n),[4 5 1 2 6 3]);
                y = X_sug.*B;
                f = permute(repmat(F_norm,1,1,layer.outputs,1,1,c+1),[1 2 3 6 4 5]);
                Y = reshape(sum(sum(y.*f,4),6),h,w,layer.outputs,n);
            else
                B = permute(repmat(layer.b,1,1,h,w,n),[3 4 1 5 2]);
                Y = reshape(sum(repmat(F_norm,1,1,layer.outputs,1,1).*B,5),h,w,layer.outputs,n);
            end
            
        end
        
        function Y = forward(layer,X)
            x_M = X(1,:,:,:);
            x_S = X(2,:,:,:);
            
            [h,w,c,n] = size(x_M);
            C = permute(repmat(layer.centers,1,1,h,w,n),[3 4 1 5 2]);
            S = permute(repmat(layer.sigma,1,1,h,w,n),[3 4 1 5 2]);
            
            MU = exp( -(C-repmat(x_M,1,1,1,1,layer.rules)).^2 ./ (2* (S.^2+(repmat(x_S,1,1,1,1,layer.rules).^2) ) ) );
            
            %% Calculate The Firing Levels
            
            F = prod(MU,3); F_norm = F./sum(F,5); %%NORMALIZED FIRING LEVEL VECTOR
            
            %% Calculate The Output
            if layer.type == "sugeno"
                X_sug = permute(repmat(cat(3,x_M,ones(h,w,1,n)),1,1,1,1,layer.outputs,layer.rules),[1 2 5 3 4 6]);
                B = permute(repmat(layer.b,1,1,1,h,w,n),[4 5 1 2 6 3]);
                y = X_sug.*B;
                f = permute(repmat(F_norm,1,1,layer.outputs,1,1,c+1),[1 2 3 6 4 5]);
                Y = reshape(sum(sum(y.*f,4),6),h,w,layer.outputs,n);
            else
                B = permute(repmat(layer.b,1,1,h,w,n),[3 4 1 5 2]);
                Y = reshape(sum(repmat(F_norm,1,1,layer.outputs,1,1).*B,5),h,w,layer.outputs,n);
            end
            
        end
        
    end
end