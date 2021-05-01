function [q_emission] = est_emission_gauss(y,X,B)
N = size(y,1); N_filters=size(X,2);
yN = repmat(y,1,N_filters); diff= yN-X;
log_q = -0.5*dot(diff,B*diff);
log_q = log_q - mean(log_q(:));
q_emission = exp(log_q);
q_emission = q_emission/sum(q_emission);
end

