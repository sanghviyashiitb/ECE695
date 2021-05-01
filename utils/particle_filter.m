function x_pred = particle_filter(y, N_filters, q_initial, q_trans, q_emission, emission_type)

x0 = q_initial{1}; Sigma = q_initial{2}; 
A = q_trans{1}; b = q_trans{2}; Sigma0 = q_trans{3}; 
if strcmp(emission_type,'gaussian')
    C = q_emission{1}; d = q_emission{2}; B1 = q_emission{3};
end
if strcmp(emission_type,'poisson')
    C = q_emission{1}; d = q_emission{2}; alpha_poiss = q_emission{3};
end
if strcmp(emission_type,'bernoulli')
    Nb = q_emission{1}; 
    C = q_emission{2}; d = q_emission{3}; 
end

D = size(x0,1);
T = size(y,2); 
%Initialize weights and "centers" for t=0
x_p = mvnrnd(x0', Sigma, N_filters)';
alpha = ones(1,N_filters)/N_filters;
x_pred = zeros(D,T);
for t=1:T
    % Time Update
    % Uses q_trans to sample from the distribution
    x_p_prev=x_p;
    alpha_prev=alpha;
    
%     gm = gmdistribution( (A*x_p_prev + b).', Sigma0, alpha_prev');
%     x_p = random(gm, N_filters).';
     
    x_p = sample_GMM(alpha_prev, A*x_p_prev+b, Sigma0, N_filters);
    
    
    % Measurement Update
    % Uses q_emission to estimate new weights
    y_t =y(:,t);
    if strcmp(emission_type,'gaussian')
        alpha =  est_emission_gauss(y_t, C*x_p + d, B1);
    end
    if strcmp(emission_type,'poisson')
        alpha =  est_emission_poiss(y_t./alpha_poiss,  exp(C*x_p + d) );
    end
    if strcmp(emission_type,'bernoulli')
        alpha =  est_emission_bern(y_t,  Nb, sigmoid(C*x_p + d) );
    end
    % Estimate mean of distribution as the estimate of x_t
    x_pred(:,t) = sum(repmat(alpha,D,1).*x_p,2);
end



end

