function [q_bern] = est_emission_bern(Y, Nb, p)
log_q = sum(Y.*log(p) + (Nb-Y).*log(1-p), 1) ;
log_q = log_q - max(log_q);
q_bern = exp(log_q);
q_bern = q_bern/sum(q_bern(:));
end
