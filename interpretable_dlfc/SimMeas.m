%%
C = C_learned';
S = S_learned';

[num_rule,num_dim] = size(C);
combs = nchoosek(1:num_rule,2);
[num_combs,~] = size(combs);
RHO = zeros(num_combs,num_dim); 

for ii = 1:num_dim
    for jj = 1:num_combs
        RHO(jj,ii) = simtaker(C(combs(jj,1),ii),S(combs(jj,1),ii),...
            C(combs(jj,2),ii),S(combs(jj,2),ii));
    end
end