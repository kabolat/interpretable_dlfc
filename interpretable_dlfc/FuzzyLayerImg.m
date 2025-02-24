classdef FuzzyLayerImg < nnet.layer.Layer
    
    properties (Learnable)
        b
        centers
        sigma
    end
    
    properties
        rules
        inputs
        type
        outputs
    end
    
    methods
        function layer = FuzzyLayerImg(N,name,M,R,Type)
            
            % N: Number of Outputs
            
            layer.Name = name;
            layer.type = Type;
            layer.outputs = N;
            layer.rules = R;
            layer.inputs = M;
            
            layer.centers = repmat((linspace(-1,1,R)),M,1);
            layer.sigma = ones(M,R);
            
            if Type == "sugeno"
                layer.b = randn(N,layer.inputs+1,layer.rules);
            else
                layer.b = randn(N,layer.rules);
            end
        end
        
        
        function Y = predict(layer,X)
            
            [h,w,c,n] = size(X);
            C = permute(repmat(layer.centers,1,1,h,w,n),[3 4 1 5 2]);
            S = permute(repmat(layer.sigma,1,1,h,w,n),[3 4 1 5 2]);
            
            %% Calculate The Firing Levels
            MU = exp(-(repmat(X,1,1,1,1,layer.rules)-C).^2./(2*S.^2));
            F = prod(MU,3); F_norm = F./sum(F,5); %%NORMALIZED FIRING LEVEL VECTOR
            
            %% Calculate The Output
            if layer.type == "sugeno"
                X_sug = permute(repmat(cat(3,X,ones(h,w,1,n)),1,1,1,1,layer.outputs,layer.rules),[1 2 5 3 4 6]);
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
            
            [h,w,c,n] = size(X);
            C = permute(repmat(layer.centers,1,1,h,w,n),[3 4 1 5 2]);
            S = permute(repmat(layer.sigma,1,1,h,w,n),[3 4 1 5 2]);
            
            %% Calculate The Firing Levels
            MU = exp(-(repmat(X,1,1,1,1,layer.rules)-C).^2./(2*S.^2));
            F = prod(MU,3); F_norm = F./sum(F,5); %%NORMALIZED FIRING LEVEL VECTOR
            
            %% Calculate The Output
            if layer.type == "sugeno"
                X_sug = permute(repmat(cat(3,X,ones(h,w,1,n)),1,1,1,1,layer.outputs,layer.rules),[1 2 5 3 4 6]);
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