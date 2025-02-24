classdef modifiedFuzzyLayer < nnet.layer.Layer
    
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
        function layer = modifiedFuzzyLayer(N,name,mfParams,Type)
            
            N: Number of Outputs
            
            layer.Name = name;
            Initialize layer weights.
            layer.centers = mfParams(:,:,2);
            layer.sigma  = mfParams(:,:,1);
            
            [layer.inputs,layer.rules,~] = size(mfParams);
            layer.type = Type;
            layer.outputs = N;
            
            if Type == "sugeno"
                layer.b = randn(layer.inputs+1,layer.rules,N);
            else
                layer.b = randn(layer.rules,N);
            end
        end
        
              
        
        
        
        function Y = predict(layer,X)
            
            total_batch = size(X,4);
                        
            Y = repmat(0*X(1,1,1,:),layer.outputs,1);
            
            for sample_no = 1:total_batch
                X_obs = squeeze(X(:,:,:,sample_no));
                
                %% Normalization
                %                 scl = X_obs-(max(X_obs)+min(X_obs))/2;
                %                 X_obs = scl./max(scl);
                
                %% Calculate The Firing Levels
                MU = exp(-(permute(repmat(X_obs,1,layer.rules),[2 1 3 4])-layer.centers').^2./(2*layer.sigma'.^2));
                
                F = prod(MU,2); F_norm = F./sum(F); %%NORMALIZED FIRING LEVEL VECTOR
                
                %% Celculate The Output
                
                if size(layer.b,3)~=1
                    for n = 1:layer.outputs
                        Y(n,1,1,sample_no) = sum(F_norm.*...
                            sum(squeeze(layer.b(:,n,:)).*repmat([1 X_obs'],layer.rules,1),2));
                    end
                else
                    for n = 1:layer.outputs
                        Y(n,1,1,sample_no) = sum(F_norm.*layer.b(:,n));
                    end
                end
                
            end
            Y = permute(Y,[3,2,1,4]);
        end
        
        function Y = forward(layer,X)
            
            total_batch = size(X,4);
            
            [R,N,~]=size(layer.b);
            
            %% Take The Parameters
            C = layer.centers; %%CENTER OF GAUSSIANS MATRIX (R-by-M)
            S = layer.sigma; %%VARIANCE OF GAUSSIANS MATRIX (R-by-M)
            B = layer.b; %%CONSEQUENTS OF THE RULES (R-by-N(-by-M+1))
            
            Y = repmat(0*X(1,1,1,:),N,1);
            
            for sample_no = 1:total_batch
                X_obs = squeeze(X(:,:,:,sample_no));
                
                %% Normalization
                %                 scl = X_obs-(max(X_obs)+min(X_obs))/2;
                %                 X_obs = scl./max(scl);
                
                %% Calculate The Firing Levels
                MU = exp(-(permute(repmat(X_obs,1,R),[2 1 3 4])-C).^2./(2*S.^2));
                
                F = prod(MU,2); F_norm = F./sum(F); %%NORMALIZED FIRING LEVEL VECTOR
                
                %% Celculate The Output
                
                if size(B,3)~=1
                    for n = 1:N
                        Y(n,1,1,sample_no) = sum(F_norm.*...
                            sum(squeeze(B(:,n,:)).*repmat([1 X_obs'],R,1),2));
                    end
                else
                    for n = 1:N
                        Y(n,1,1,sample_no) = sum(F_norm.*B(:,n));
                    end
                end
                
            end
            Y = permute(Y,[3,2,1,4]);
        end
    end
    
end