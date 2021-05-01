function [q_poisson] = Poisson_Estimation(X, Y, train_alpha)
%Estimate C,d s.t y_n = P(C*x_n +d_n) using IRLS
% Concatenate 1 to the 
d = size(X,1); N = size(X,2); 
T = size(Y,1);
X = cat(1, X, ones(1,N));
C = 0.001*randn( T,d+1);
if train_alpha
    alpha = 10*ones(T,1);
else
    alpha = ones(T,1);
end
max_iters = 50;
step_size = 0.3;

disp('Estimating Poisson emission parameters using IRLS');
for iterations=1:max_iters
    mu = exp(C*X);
    Yhat = Y./repmat(alpha,1,N);
    grad_res = X*(Yhat-mu)';
    diff_C = C*0;
    for n = 1:T
        % Calculating hessian and gradient for each row separately
        grad_c_n  = grad_res(:,n);
        w = mu(n,:);
        hessian_c_n = X*diag(w)*X';
        diff_c_n = ((hessian_c_n)^(-1))*grad_c_n;
        diff_C(n,:) = diff_c_n';
%         diff_C(n,:) = grad_c_n';
    end 
    if train_alpha
        mu_hat = exp(C*X);
        for n = 1:T
            m_n = mu_hat(n,:); y_n = Y(n,:);
            alpha(n,1) =  (y_n*m_n')*((m_n*m_n')^(-1));
        end
    end     
    if mod(iterations,500) == 0
        step_size=step_size*0.1;
    end
    C= C + step_size*diff_C;
    diff_norm = norm(  repmat(alpha,1,N).*exp(C(:,1:d)*X(1:d,:)+ C(:,d+1))-Y,'fro')^2/norm(Y,'fro')^2;
    disp(['Iterations: ',num2str(iterations),', Relative Diff: ',num2str(diff_norm)]);
    
    if diff_norm < 1e-2
        break;
    end
        
q_poisson ={};
q_poisson{1} = C(:,1:d); q_poisson{2} = C(:,d+1); q_poisson{3} = alpha;
    
end

