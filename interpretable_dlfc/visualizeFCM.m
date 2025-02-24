function mfparams = visualizeFCM(z, C, U, opt)

maxZ = max(z);
minZ = min(z);

[numClst,dims] = size(C);

mfparams = zeros(numClst,dims,2);

for ii = 1:dims
    subplot(dims,1,ii)
    hold on
    for jj = 1:numClst
%          stem(z(:,ii)',U(jj,:))
        sigma = invgaussmf4sigma(z(:,ii), U(jj,:)', C(jj,ii));
        mfparams(jj,ii,:) = [sigma, C(jj,ii)];
        if opt == 1
        plot(linspace(minZ(ii),maxZ(ii),1000), exp(-(linspace(minZ(ii),maxZ(ii),1000)-C(jj,ii)).^2./(2*sigma^2)),...
            'LineWidth', 3);
        end
    end
    hold off
end


end