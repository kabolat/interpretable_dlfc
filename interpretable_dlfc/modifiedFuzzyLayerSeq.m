classdef modifiedFuzzyLayerSeq < nnet.layer.Layer
    
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
        function layer = modifiedFuzzyLayerSeq(N,name,mfParams,Type)
            
            % N: Number of Outputs
            
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
            
            [c,n,s] = size(X);
            C = permute(repmat(layer.centers,1,1,n,s),[1 3 4 2]);
            S = permute(repmat(layer.sigma,1,1,n,s),[1 3 4 2]);
            
            %% Calculate The Firing Levels
            MU = exp(-(repmat(X,1,1,1,layer.rules)-C).^2./(2*S.^2));
            F = prod(MU,1); F_norm = F./sum(F,4); %%NORMALIZED FIRING LEVEL VECTOR
            
            %% Calculate The Output
            if layer.type == "sugeno"
                X_sug = permute(repmat(cat(1,X,ones(1,n,s)),1,1,1,layer.outputs,layer.rules),[4 1 2 3 5]);
                B = permute(repmat(layer.b,1,1,1,n,s),[1 2 4 5 3]);
                y = X_sug.*B;
                f = permute(repmat(F_norm,layer.outputs,1,1,1,c+1),[1 5 2 3 4]);
                Y = squeeze(sum(sum(y.*f,2),5));
            else
                B = permute(repmat(layer.b,1,1,n,s),[1 3 4 2]);
                Y = squeeze(sum(repmat(F_norm,layer.outputs,1,1,1).*B,4));
            end
            
        end
        
%         function Y = forward(layer,X)
%             
%             [c,n,s] = size(X);
%             C = permute(repmat(layer.centers,1,1,n,s),[1 3 4 2]);
%             S = permute(repmat(layer.sigma,1,1,n,s),[1 3 4 2]);
%             
%             %% Calculate The Firing Levels
%             MU = exp(-(repmat(X,1,1,1,layer.rules)-C).^2./(2*S.^2));
%             F = prod(MU,1); F_norm = F./sum(F,4); %%NORMALIZED FIRING LEVEL VECTOR
%             
%             %% Calculate The Output
%             if layer.type == "sugeno"
%                 X_sug = permute(repmat(cat(1,X,ones(1,n,s)),1,1,1,layer.outputs,layer.rules),[4 1 2 3 5]);
%                 B = permute(repmat(layer.b,1,1,1,n,s),[1 2 4 5 3]);
%                 y = X_sug.*B;
%                 f = permute(repmat(F_norm,layer.outputs,1,1,1,c+1),[1 5 2 3 4]);
%                 Y = squeeze(sum(sum(y.*f,2),5));
%             else
%                 B = permute(repmat(layer.b,1,1,n,s),[1 3 4 2]);
%                 Y = squeeze(sum(repmat(F_norm,layer.outputs,1,1,1).*B,4));
%             end
%             
%         end
        
    end
end