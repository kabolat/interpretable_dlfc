function meas = simtaker(c1,s1,c2,s2)

x = linspace(-5,5,500);

g1 = @(x1) exp(-(x1-c1).^2/(2*s1^2));
g2 = @(x2) exp(-(x2-c2).^2/(2*s2^2));

min_area = trapz(x,min([g1(x);g2(x)]));
max_area = trapz(x,max([g1(x);g2(x)]));

meas = min_area/max_area;

end