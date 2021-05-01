function q_state = State_Model_Estimation(x)
T = size(x,2); 
x_t1 = circshift(x, 1, 2);
x_t = x(:,2:T); x_t1 = x_t1(:,2:T);

X = x_t*x_t'/(T-1); Y = x_t*x_t1'/(T-1);
x_m = mean(x_t,2); x_t1_m = mean(x_t1,2);

A = (Y-x_m*x_t1_m')*((X - x_m*x_m')^(-1));
b = x_t1_m - A*x_m;
res = A*x_t+b-x_t1;
Sigma0 = (res*res')/(T-2);


q_state = {};
q_state{1} = A; q_state{2} = b; q_state{3} = Sigma0;

end

