clc; clear; close all
%% loading data

filePath = 'C:\Users\akbaris\Saia\Papers\[2024-Automatica] [Submitted] - Lb Dropout DNN\Simulation\HeatMap_n\matlab';


data = load(filePath, 'n_values', 'step_size', 'e_rms_results', 'ftilde_rms_results', 'u_rms_results');

n = data.n_values;
n = n';
step_size = data.step_size;
e_rms_results = data.e_rms_results;
ftilde_rms_results = data.ftilde_rms_results;
u_rms_results = data.u_rms_results;





%% plot 
subplot(3, 1, 1);
plot(n, e_rms_results','-o', 'linewidth', 1.5);
%set(gca, 'YLim', [0.7 1.45]);
set(gca, 'XLim', [0.8 10.2], 'XTick', 0:1:10,'XTickLabel', 0:1:10);
ylabel('RMS $||e||$','Fontsize',16,'Interpreter','latex')

subplot(3, 1, 2);
plot(n, ftilde_rms_results','-o', 'linewidth', 1.5);
%set(gca, 'YLim', [8 48]);
set(gca, 'XLim', [0.8 10.2], 'XTick', 0:1:10,'XTickLabel', 0:1:10);
ylabel('RMS $||f(x)-\widehat{\Phi}||$','Fontsize',16,'Interpreter','latex')

subplot(3, 1, 3);
plot(n, u_rms_results','-o', 'linewidth', 1.5);
%set(gca, 'YLim', [25 54]);
set(gca, 'XLim', [0.8 10.2], 'XTick', 0:1:10,'XTickLabel', 0:1:10);
xlabel('$\delta t$ (sec)','Fontsize',16,'Interpreter','latex');
ylabel('RMS $||u||$','Fontsize',16,'Interpreter','latex')