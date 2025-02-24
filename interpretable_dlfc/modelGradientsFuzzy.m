function [grad, loss] = modelGradientsFuzzy(encoderNet, fuzzyNet, x, y)
%Parametrization Trick ile sample al
[z, zMean, zLogvar] = sampling(encoderNet, x);

%z -> Decoder -> Sigmoid -> xPred (x'ler 0-1 arasinda)
yPred = forward(fuzzyNet,z);
yPred = squeeze(softmax(yPred));

%Tanimlanan layerlarda output layer yok. Son iki layer sigmoid ve ELBO
%olarak burada tanimlandi.

loss = crossentropy(yPred,y);

%dlgradient inputlari dlarray olduklari ve "trace"te olduklari icin
%dogrudan icine yazilabildi. Butun inputlar skaler. Outputlarin ilki
%decoderin gradyanlari, ikincisi encoderin. 
%Bu sekilde kullanabilmek icin modelGradients fonksiyonun dlfeval icinde
%olmasi lazim. obur turlu decoderNet ve encoderNet'i traced almiyor

[grad] = dlgradient(loss,fuzzyNet.Learnables);
end