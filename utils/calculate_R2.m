function [R2] = calculate_R2(x_true,x_pred)
% T = size(x_true,2); Tend = floor(0.8*T);
% x_true = x_true(1:2,Tend:end); x_pred = x_pred(1:2,Tend:end); 

x_true = x_true(1:2,:); x_pred = x_pred(1:2,:); 
xhat = mean(x_true,2);
err = (norm(x_true-x_pred,'fro')^2)/(norm(xhat-x_true,'fro')^2);
R2 = 1-err;
end

