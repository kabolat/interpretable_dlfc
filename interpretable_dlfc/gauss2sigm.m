function [a, c_s] = gauss2sigm(c_g,s,opt)

if opt == 1 %%left
    c_s = c_g+sqrt(-2*s^2*log(.5));
elseif opt == 2 %%right
    c_s = c_g-sqrt(-2*s^2*log(.5));
end

a = -2*(c_s-c_g)/(s^2);

end