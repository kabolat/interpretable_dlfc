%% ELBO(Evidence Lower BOund)
function [elbo, reconstructionLoss, KL] = ELBOloss(x, xPred, zMean, zLogvar,Beta)
%MSE
squares = (xPred-x).^2;
reconstructionLoss  = mean(squares, [1,2,3]); %Sadece H,W ve C toplamak icin [1,2,3]
%reconstructionLoss -> 1x1x1xminiBatchSize bir vektor
reconstructionLoss = mean(reconstructionLoss);

%KL Divergence
%Latent Space dagilimim Gauss'tan ne kadar farkli
KL = mean(-1*mean(1 - exp(zLogvar) - zMean.^2 + zLogvar, 1));

elbo = reconstructionLoss + Beta*KL;
 end