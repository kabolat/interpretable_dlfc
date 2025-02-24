function [KL] = KL_Loss(zMean, zSigma)
zLogvar = 2*log(zSigma);
KL = -1*mean(1 - exp(zLogvar) - zMean.^2 + zLogvar, 1);
 end