function [infGrad, genGrad1, genGrad2, z, zMean, zLogvar, xPred, loss, rloss, KL] = modelGradients(encoderNet, decoder1Net, decoder2Net, x, Beta)
%Parametrization Trick ile sample al
[z, zMean, zLogvar] = sampling(encoderNet, x);

%z -> Decoder -> Sigmoid -> xPred (x'ler 0-1 arasinda)
xx = forward(decoder1Net,z);
[chn,btch] = size(xx);
xx = dlarray(reshape(xx,4,4,32,btch),'SSCB');
xPred = 256*sigmoid(forward(decoder2Net, xx));

%Tanimlanan layerlarda output layer yok. Son iki layer sigmoid ve ELBO
%olarak burada tanimlandi.
[loss, rloss, KL] = ELBOloss(x, xPred, zMean, zLogvar, Beta);

%dlgradient inputlari dlarray olduklari ve "trace"te olduklari icin
%dogrudan icine yazilabildi. Butun inputlar skaler. Outputlarin ilki
%decoderin gradyanlari, ikincisi encoderin. 
%Bu sekilde kullanabilmek icin modelGradients fonksiyonun dlfeval icinde
%olmasi lazim. obur turlu decoderNet ve encoderNet'i traced almiyor

[genGrad2, genGrad1, infGrad] = dlgradient(loss, decoder2Net.Learnables,decoder1Net.Learnables, ...
    encoderNet.Learnables);
end