function [centers, U] = latent_fcm(z,opts)

[centers,U] = fcm(z,opts(1),opts(2:end));

end