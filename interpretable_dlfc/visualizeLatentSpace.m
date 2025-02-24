function z = visualizeLatentSpace(XTest, YTest, encoderNet)
z = sampling(encoderNet, XTest);

z = stripdims(z);
z = gather(extractdata(z));
z = squeeze(z)';
z = double(z);

[~,score] = pca(z);

c = parula(10);     %RGB icin

%scatter3(score(:,1),score(:,2),score(:,3),[],c(double(YTest),:));
scatter3(z(:,1),z(:,2),z(:,3),[],c(double(YTest),:));
axis equal
xlabel("Z(1)")
ylabel("Z(2)")
zlabel("Z(3)")
cb = colorbar; cb.Ticks = 0:(1/9):1; cb.TickLabels = string(0:9);

end