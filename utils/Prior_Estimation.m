function q_initial = Prior_Estimation(x)
T = size(x,2);
b = mean(x,2);
Sigma0 = (x-b)*(x-b)'/(T-1);
q_initial = {};
q_initial{1} = b; q_initial{2} = Sigma0;
end

