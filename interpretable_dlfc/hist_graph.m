clc; clear all; close all;

%%

load anfisbeta001dim10_newArch_clst3_mean.mat

[sorted_KL, indx_KL] = sort(KL,'descend');
cum_KL = cumsum(sorted_KL);
sum_KL = cum_KL(end);
cum_KL = cum_KL/sum_KL;

thr = 3;

ampt_z_indx = indx_KL(1:thr);

numInputs = length(ampt_z_indx);

z_fcm_trainn = z_fcm_train(:,:,ampt_z_indx,:);

mfparamss = mfparams(ampt_z_indx,:,:);

C_learned = squeeze(mfparamss(:,:,2));
S_learned = squeeze(mfparamss(:,:,1));

figure
subplot 231
histogram(z_fcm_trainn(:,:,1,:))
title('z_9')
xlim([-5 5])

subplot 232
histogram(z_fcm_trainn(:,:,2,:))
title('z_1')
xlim([-5 5])

subplot 233
histogram(z_fcm_trainn(:,:,3,:))
title('z_8')
xlim([-5 5])

x = linspace(-5,5,1000);
for ii = 1:numInputs
    subplot(2,numInputs,ii+3)
    hold on
    for jj = 1:numRules
        plot(x, exp(-(x-C_learned(ii,jj)).^2./(2*S_learned(ii,jj)^2)),...
            'LineWidth', 3);
    end
    hold off
    xlabel(['KL Loss=' num2str(sorted_KL(ii))])
end

%%

figure
c = parula(10);     %RGB icin

z = squeeze(z_fcm_trainn)';

scatter3(z(:,1),z(:,2),z(:,3),[],c(double(YTrain(1e4+1:end)),:));
axis equal
xlabel("Z(9)")
ylabel("Z(1)")
zlabel("Z(8)")
cb = colorbar; cb.Ticks = 0:(1/9):1; cb.TickLabels = string(0:9);
