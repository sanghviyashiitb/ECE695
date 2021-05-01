function [r2] = sample_GMM(pi,C,Sigma,N)
% K = number of classes, N = number of samples, d = dimension
% C = [d, K]
% mixture coin tosses
d = size(C,1); K = size(C,2);
mix_p = mnrnd(1,pi,N); % [N,K] mixture probabilities
m = C*mix_p'; % [d,N] N dX1 column vectors representing mean of each sample


r1 = mvnrnd(zeros(1,d), Sigma, N)';  % d X N random vectors drawn from N(0, Sigma)
r2 = m + r1;  % d X N random vectors drawn from N(m_n, Sigma)
end

