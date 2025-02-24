function [zSampled, zMean, zvar] = visualizeLatentSpaceMeanVar(XTest, YTest, encoderNet,dim,opt)
if dim == 2
    %% 2D Latent
    [zSampled, zMean, zLogvar] = sampling(encoderNet, XTest);
    
    zMean = stripdims(zMean);
    zMean = gather(extractdata(zMean));
    zMean = squeeze(zMean)';
    
    zLogvar = stripdims(zLogvar);
    zLogvar = gather(extractdata(zLogvar));
    zLogvar = squeeze(zLogvar)';
    zvar = exp(.5*zLogvar);
    
    zSampled = stripdims(zSampled);
    zSampled = gather(extractdata(zSampled));
    zSampled = squeeze(zSampled)';
    
    if opt==1
        c = parula(10);     %RGB icin
        f1 = figure;
        figure(f1)
        title("Latent space")
        
        
        subplot(1,3,1);
        scatter(zMean(:,1),zMean(:,2),[],c(double(YTest),:));
        axis equal
        xlabel("Z_m_u(1)")
        ylabel("Z_m_u(2)")
        zlabel("Z_m_u(3)")
        
        subplot(1,3,2);
        scatter(zvar(:,1),zvar(:,2),[],c(double(YTest),:));
        xlabel("Z_v_a_r(1)")
        ylabel("Z_v_a_r(2)")
        zlabel("Z_v_a_r(3)")
        axis equal
        
        subplot(1,3,3);
        scatter(zSampled(:,1),zSampled(:,2),[],c(double(YTest),:));
        xlabel("Z_s_m_p(1)")
        ylabel("Z_s_m_p(2)")
        zlabel("Z_s_m_p(3)")
        cb = colorbar;  cb.Ticks = 0:(1/9):1; cb.TickLabels = string(0:9);
        axis equal
    end
else
    
    %% 3D Latent
    [zSampled, zMean, zLogvar] = sampling(encoderNet, XTest);
    
    zMean = stripdims(zMean);
    zMean = gather(extractdata(zMean));
    zMean = squeeze(zMean)';
    
    zLogvar = stripdims(zLogvar);
    zLogvar = gather(extractdata(zLogvar));
    zLogvar = squeeze(zLogvar)';
    zvar = exp(.5*zLogvar);
    
    zSampled = stripdims(zSampled);
    zSampled = gather(extractdata(zSampled));
    zSampled = squeeze(zSampled)';
    
    if opt == 1
        c = parula(10);     %RGB icin
        f1 = figure;
        figure(f1)
        title("Latent space")
        
        subplot(1,3,1);
        scatter3(zMean(:,1),zMean(:,2),zMean(:,3),[],c(double(YTest),:));
        axis equal
        xlabel("Z_m_u(1)")
        ylabel("Z_m_u(2)")
        zlabel("Z_m_u(3)")
        
        subplot(1,3,2);
        scatter3(zvar(:,1),zvar(:,2),zvar(:,3),[],c(double(YTest),:));
        xlabel("Z_v_a_r(1)")
        ylabel("Z_v_a_r(2)")
        zlabel("Z_v_a_r(3)")
        axis equal
        
        subplot(1,3,3);
        scatter3(zSampled(:,1),zSampled(:,2),zSampled(:,3),[],c(double(YTest),:));
        xlabel("Z_s_m_p(1)")
        ylabel("Z_s_m_p(2)")
        zlabel("Z_s_m_p(3)")
        cb = colorbar;  cb.Ticks = 0:(1/9):1; cb.TickLabels = string(0:9);
        axis equal
    end
end
end