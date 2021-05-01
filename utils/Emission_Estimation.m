function q_emmision = Emission_Estimation(x, y)
T = size(x,2); 
% y_t = N(Cx_t + d, Sigma_1), Estimating C, d and Sigma_1
X = x*x'/T; Y = y*x'/T;
x_m = mean(x,2); y_m = mean(y,2);

C = (Y-y_m*x_m')*((X - x_m*x_m')^(-1));
d = y_m - C*x_m;
res = C*x+d-y;
Sigma1 = (res*res')/(T-1);

% Pseudoinverse
[U1, S1, V1] = svd(Sigma1);
s1 = diag(S1);
S11 = diag((s1>1e-6).*(1./(s1+1e-6)));

q_emmision = {};
q_emmision{1}=C; 
q_emmision{2}=d; 
q_emmision{3} = U1*S11*V1';
q_emmision{4} = Sigma1;

end

