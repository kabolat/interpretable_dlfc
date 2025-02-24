function [zSampled, zMean, zLogvar] = sampling(encoderNet, x)
%x: MiniBatch

compressed = forward(encoderNet,x);     %Encoder Forward-Pass
d = size(compressed,1)/2;               %Mean ve Varyans boyutlari esit
% [2*d, miniBatchSize] = size(compressed)
zMean = compressed(1:d,:);              %Ilk d eleman Mean
zLogvar = compressed(1+d:end,:);        %Son d eleman logVaryans

sz = size(zMean);
epsilon = randn(sz);                    %Batch'teki her eleman icin randn
sigma = exp(.5 * zLogvar);              %logVaryans -> Standart Sapma
z = epsilon .* sigma + zMean;
z = reshape(z, [1,1,sz]);               %Kanal Sayisi:2*d, Batch Sayisi:miniBatchSize
zSampled = dlarray(z, 'SSCB');
end