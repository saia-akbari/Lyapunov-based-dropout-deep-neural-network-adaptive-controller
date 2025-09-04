clc;
clear;
close all;

step_size = 0.01;
simtime = 10;
time_length = simtime / step_size;
x=[5;1;-5; 1; 3];
ke=15;
ks=0.001;
Gamma = 100; % Adaptation Gain

% DNN Parameters
s = 5;   % Size of the State; 
k = 25;   % Total Number of Hidden Layers 
L = 10;  % Width of Each Hidden Layer 
L_in = (s);
L_out = s; 
L_vec = (L_out + L_in + (k - 1) * L) * L;

vecV = 10 * rand(L_vec, 1);

% Define the range for the second element of n
n_values = 1:L;  % Assuming s is in the range 1 to 8
deltat = 20;

% Preallocate arrays to store results
e_rms_results = zeros(length(n_values), 1);
ftilde_rms_results = zeros(length(n_values), 1);
u_rms_results = zeros(length(n_values), 1);

for i = 1:length(n_values)
    n = [s+1 n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) ...
        n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) ...
        n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) ...
        n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) n_values(i) n_values(i)];
    

    [e_RDNN, ftilde_RDNN, u_list_RDNN, ~, ~, ~] = ...
        RDNN_func(k, n, L, s, 2, deltat, "tanh", L_in, L_out, L_vec, vecV, step_size, simtime, x, ke, ks, Gamma);

    e_rms_results(i) = norm(rms(e_RDNN'));
    ftilde_rms_results(i) = norm(rms(ftilde_RDNN'));
    u_rms_results(i) = norm(rms(u_list_RDNN'));
end

% Save the data
save('RDNN_results.mat', 'n_values', 'e_rms_results', 'ftilde_rms_results', 'u_rms_results');

% Create heatmaps
figure;

subplot(3, 1, 1);
imagesc(1, n_values, e_rms_results');
%title('e\_rms\_RDNN with respect to the values of the second element of n');
colorbar;
set(gca, 'YTick', []);
ylabel('RMS $||e||$','Fontsize',16,'Interpreter','latex')

subplot(3, 1, 2);
imagesc(1, n_values, ftilde_rms_results');
%title('ftilde\_rms\_RDNN with respect to the values of the second element of n');
colorbar;
set(gca, 'YTick', []);
ylabel('RMS $||f(x)-\widehat{\Phi}||$','Fontsize',16,'Interpreter','latex')

subplot(3, 1, 3);
imagesc(1, n_values, u_rms_results');
%title('u\_rms\_RDNN with respect to the values of the second element of n');
colorbar;
set(gca, 'YTick', []);
xlabel('Number of neurons that are not dropped out','Fontsize',16,'Interpreter','latex');
ylabel('RMS $||u||$','Fontsize',16,'Interpreter','latex')