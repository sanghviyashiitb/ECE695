function [x_pred] = kalman_filter(y, q_initial, q_trans, q_emission)
m0 = q_initial{1}; Sigma = q_initial{2}; D = size(m0,1);
A = q_trans{1}; b = q_trans{2}; Sigma0 = q_trans{3}; 
C = q_emission{1}; d = q_emission{2}; B1 = q_emission{3}; 

T = size(y,2); N = size(C,1);
x_pred = zeros(D,T);

% Convention defined here
% \hat{x}_{t|t-1} = xt_1,   \hat{x}_{t|t} = xt
% P_{t|t-1} = Pt_1,         P_{t|t} = Pt;
xt1 = m0; Pt1 = Sigma;
for t=1:T
    % Gain Factor: K_n
    Kt = eye(N) - C*((Pt1^(-1) + C'*B1*C)^(-1))*C'*B1; 
    Kt = Pt1*C'*B1*Kt;
    
    % Measurement Update
    xt = xt1 + Kt*(y(:,t) - d - C*xt1);
    Pt = (eye(D) - Kt*C)*Pt1;
    x_pred(:,t) = xt;
    
    % Time Update for next step i.e. looking one step ahead
    xt1 = A*xt + b;
    Pt1 = A*Pt*A^T + Sigma0;   
end

end

