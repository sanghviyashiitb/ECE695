%%%
% Script comparing different Kalman and Particl Filter under same inference
% model
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

disp(['Sampling Time: ',num2str(SamplingTime*1000),'ms']);

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


q_initial = Prior_Estimation(xb_train);
q_state = State_Model_Estimation(xb_train);
q_emission = Emission_Estimation(xb_train, yb_train);

disp('Running Kalman  Filter');
tic;
xb_kf = kalman_filter(yb_test, q_initial, q_state, q_emission);
toc;
SNR = -10*log10(1-calculate_R2(xb_test, xb_kf));
disp(['Kalman Filter, SNR: ',num2str(SNR)]);

disp('Running Particle Filter, Gaussian Emission');
tic;
xb_pf = particle_filter(yb_test, 800, q_initial, q_state, q_emission, 'gaussian');
toc;
SNR = -10*log10(1-calculate_R2(xb_test, xb_pf));
disp(['Particle Filter, SNR: ',num2str(SNR)]);

figure(1); 
N1 = 300; N2 = 320;
subplot(1,2,1); hold on;
plot(xb_test(1,N1:N2),'r'); plot(xb_kf(1,N1:N2),'g--'); plot(xb_pf(1,N1:N2),'b-.'); 
ylabel('X Coordinate'); legend('True', 'Kalman Filter', 'Particle Filter');
hold off;

subplot(1,2,2); hold on;
plot(xb_test(2,N1:N2),'r'); plot(xb_kf(2,N1:N2),'g--'); plot(xb_pf(2,N1:N2),'b-.');
ylabel('Y Coordinate'); legend('True', 'Kalman Filter', 'Particle Filter');
hold off;
