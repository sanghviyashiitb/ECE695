function [q_bernoulli] = Bernoulli_Estimation(X, Y, Nb)
% Estimate bernoulli distr. using IRLS. NUmber of trials "Nb" set by user
% Concatenate 1 to the 
d = size(X,1); N = size(X,2); 
T = size(Y,1);
X = cat(1, X, ones(1,N));
C = 0.001*randn(T,d+1);

max_iters = 200;
step_size = 0.1;
disp('Estimating Poisson emission parameters using IRLS');
for iterations=1:max_iters
    mu = sigmoid(C*X); mu1 = 1-mu;
    grad_res = X*(Y.*mu1 - (Nb-Y).*mu)';
    diff_C = C*0;
    for n = 1:T
        % Calculating hessian and gradient for each row separately
        grad_c_n  = grad_res(:,n);
        w = -Nb*mu1(n,:).*mu(n,:);
        hessian_c_n = -X*diag(w)*X';
        diff_c_n = ((hessian_c_n)^(-1))*grad_c_n;
        diff_C(n,:) = diff_c_n';
    end 
    if mod(iterations,500) == 0
        step_size=step_size*0.1;
    end
    C = C + step_size*diff_C;
    diff_norm = norm(Nb*sigmoid(C(:,1:d)*X(1:d,:)+ C(:,d+1))-Y,'fro')^2/norm(Y,'fro')^2;
    disp(['Iterations: ',num2str(iterations),', Relative Diff: ',num2str(diff_norm)]);
    
    if diff_norm < 1e-2
        break;
    end
end
        
q_bernoulli ={};
q_bernoulli{1} = Nb; q_bernoulli{2} = C(:,1:d); q_bernoulli{3} = C(:,d+1);
    
end
