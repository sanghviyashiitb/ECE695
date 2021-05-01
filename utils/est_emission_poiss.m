function [q_emission] = est_emission_poiss(y,mu)
log_q = -sum(mu, 1) + y'*log(mu);
log_q = log_q - max(log_q);
q_emission = exp(log_q);
q_emission = q_emission/sum(q_emission);
end

