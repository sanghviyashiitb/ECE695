%%%
% Script comparing different emission probabilities - Gaussian, Poisson,
% and Binomial. Note that this script will take a few minutes to run
%%%

clear all
close all

addpath('./utils');
disp('Loading data file');
load indy_20160411_01.mat


Freq = 24400;   % original sampling frequency
B = 128;         % binning size
SamplingTime = B/Freq;
T = size(t,1);

N = size(chan_names,1);
x = finger_pos;
y = zeros(T,N);
for idx=1:N
    spike = spikes{idx};
    if size(spike) > 0
        y(:,idx) =  histc(spike,t)';
    end
end
clear wf chan_names spikes

% Remove the z coordinate and concatenate discrete derivative as a feature
x = x(:,2:3); v = diff(x); x = [x(2:end,:), 100*v];

[xb, yb] = binning(x,y,B);
xb= xb'; yb =yb';
Tb = size(xb,2);
% Partition data into 'training' and 'test'
T1 = floor(Tb*0.5);
xb_train = xb(:,1:T1); xb_test = xb(:,T1+1:end);
yb_train = yb(:,1:T1); yb_test = yb(:,T1+1:end);
% Remove all the zero indices of Y
idx_non_zero = find( sum(yb_train,2) > 0 );
yb_train = yb_train(idx_non_zero, :); yb_test = yb_test(idx_non_zero, :); 


N_filters = 4000;
q_initial = Prior_Estimation(xb_train);
q_state = State_Model_Estimation(xb_train);

% Different Emission Probabilities
q_gauss = Emission_Estimation(xb_train, yb_train);
xb_gauss = particle_filter(yb_test, N_filters, q_initial, q_state, q_gauss, 'gaussian');
SNR = -10*log10(1-calculate_R2(xb_test, xb_gauss))


q_poisson1 = Poisson_Estimation(xb_train, yb_train, false);
xb_poiss1 = particle_filter(yb_test, N_filters, q_initial, q_state, q_poisson1, 'poisson');
SNR = -10*log10(1-calculate_R2(xb_test, xb_poiss1))

q_poisson2 = Poisson_Estimation(xb_train, yb_train, true);
xb_poiss2 = particle_filter(yb_test, N_filters, q_initial, q_state, q_poisson2, 'poisson');
SNR = -10*log10(1-calculate_R2(xb_test, xb_poiss2))

q_bern = Bernoulli_Estimation(xb_train, yb_train, 100);
xb_bern = particle_filter(yb_test, N_filters, q_initial, q_state, q_bern, 'bernoulli');
SNR = -10*log10(1-calculate_R2(xb_test, xb_bern))


figure(1); 
N1 = 300; N2 = 320;
subplot(1,2,1); hold on;
plot(xb_test(1,N1:N2),'r');     plot(xb_poiss1(1,N1:N2),'g');      
plot(xb_poiss2(1,N1:N2),'b-.');   
ylabel('X Coordinate'); legend('True', 'Poisson', 'Poisson with Scaling');
hold off;

subplot(1,2,2); hold on;
plot(xb_test(2,N1:N2),'r');     plot(xb_poiss1(2,N1:N2),'g');      
plot(xb_poiss2(2,N1:N2),'b-.');   
ylabel('Y Coordinate'); legend('True', 'Poisson', 'Poisson with Scaling');

hold off;



figure(2); 
N1 = 300; N2 = 320;
subplot(1,2,1); hold on;
plot(xb_test(1,N1:N2),'r');     plot(xb_poiss2(1,N1:N2),'g');      
plot(xb_bern(1,N1:N2),'b-.');   
ylabel('X Coordinate'); legend('True', 'Poisson with Scaling', 'Bernoulli (N = 100)');
hold off;

subplot(1,2,2); hold on;
plot(xb_test(2,N1:N2),'r');     plot(xb_poiss2(2,N1:N2),'g');      
plot(xb_bern(2,N1:N2),'b-.');   
ylabel('X Coordinate'); legend('True', 'Poisson with Scaling', 'Bernoulli (N = 100)');

hold off;