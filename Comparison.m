clc; clear; close all;

step_size=0.01;
simtime=10;
time_length=simtime/step_size;
x=[5;1;-5; 1; 3];
ke=15;
ks=0.001;
Gamma=100;  %Adaptation Gain
        

%DNN Parameters

s=5;   % Size of the State; 
k=25;   % Total Number of Hidden Layers 
L=10;  % Width of Each Hidden Layer 

L_in = (s);
L_out= s; 
L_vec= (L_out+L_in+(k-1)*L)*L;

vecV=10*rand(L_vec,1);

deltat = 20;

[e_DNN,ftilde_DNN,u_list_DNN,vecV_list_DNN,x_DNN,f_list_DNN, time_DNN] = ...
    Pruning_func(k,L,s,10,deltat,"tanh",L_in,L_out, L_vec,vecV,step_size,simtime,x,ke,ks,Gamma);
[e_RDNN,ftilde_RDNN,u_list_RDNN,vecV_list_RDNN,x_RDNN,f_list_RDNN, Phi_prime, time_RDNN]=...
    RDNN_func(k,L,s,10,deltat,"tanh",L_in,L_out, L_vec,vecV,step_size,simtime,x,ke,ks,Gamma);

e_rms_DNN=norm(rms(e_DNN'));
ftilde_rms_DNN=norm(rms(ftilde_DNN'));
u_rms_DNN=norm(rms(u_list_DNN'));
e_rms_RDNN=norm(rms(e_RDNN'));
ftilde_rms_RDNN=norm(rms(ftilde_RDNN'));
u_rms_RDNN=norm(rms(u_list_RDNN'));

cost_DNN=e_rms_DNN^2+0.01*u_rms_DNN^2;
cost_RDNN=e_rms_RDNN^2+0.01*u_rms_RDNN^2;

vecV_opt_DNN=vecV;
cost_min_DNN=cost_DNN;

vecV_opt_RDNN=vecV;
cost_min_RDNN=cost_RDNN;

cost_min_RDNN

time=(0:(time_length))*step_size;


figure(1)

    
    subplot(2,1,1)
    plot(time,vecnorm(e_DNN),time,vecnorm(e_RDNN),'LineWidth',1.5)
    ylabel('$||e||$','Fontsize',16,'Interpreter','latex', 'FontName','Times New Roman')
    legend('DNN','Dropout DNN','Interpreter','latex','Fontsize',16,'Location','NorthEast','Orientation','Horizontal', 'FontName','Times New Roman')
        
    set(gca, 'YLim', [-0.5 8.5], 'YTick', 0:2:8,...
     'YTickLabel', 0:2:8);
    set(gca, 'XLim', [-0.2 10.2], 'XTick', 0:1:10,...
     'XTickLabel', 0:1:10);
    
    subplot(2,1,2)
    plot(time,vecnorm(ftilde_DNN),time,vecnorm(ftilde_RDNN), 'LineWidth', 1.5)
    ylabel('$||f(x)-\widehat{\Phi}||$','Fontsize',16,'Interpreter','latex', 'FontName','Times New Roman')
    xlabel('Time (sec)','Fontsize',16,'Interpreter','latex', 'FontName','Times New Roman')
    legend('DNN','Dropout DNN','Interpreter','latex','Fontsize',16,'Location','NorthEast','Orientation','Horizontal', 'FontName','Times New Roman')
    
    set(gca, 'YLim', [-10 260], 'YTick', 0:50:250,...
     'YTickLabel', 0:50:250);
    set(gca, 'XLim', [-0.2 10.2], 'XTick', 0:1:10,...
     'XTickLabel', 0:1:10);

% 
% figure(2)
%     subplot(2,1,1)
%     plot(time,vecV_list_DNN)
%     ylabel('DNN Weight Estimates','Fontsize',16, 'FontName','Times New Roman')
%     xlabel('Time (s)','Fontsize',16, 'FontName','Times New Roman')
% %     ylim([-2 2])
% 
% 
%     subplot(2,1,2)
%     plot(time,vecV_list_RDNN)
%     ylabel('DDNN Weight Estimates','Fontsize',16, 'FontName','Times New Roman')
%     xlabel('Time (s)','Fontsize',16, 'FontName','Times New Roman')
%     %ylim([-1 1])
% 
% 
% 
% figure(4)
% 
% 
%     plot(time(1:1000), vecnorm(u_list_DNN), 'LineWidth',1.5)
%     ylabel('DNN Control Input','Fontsize',16, 'FontName','Times New Roman')
% 
% hold on
% 
%     plot(time(1:1000), vecnorm(u_list_RDNN), 'linewidth', 1.5)
%     ylabel('DDNN Control Input','Fontsize',16, 'FontName','Times New Roman')
%     xlabel('Time (sec)','Fontsize',16, 'FontName','Times New Roman')
% 
%     legend('DNN Controller', 'RDNN Controller')
% figure(4)
% 
%     plot(time,vecnorm(u_list_DNN),time,vecnorm(u_list_RDNN))
%     ylabel('Normalized Control Inputs')
%     xlabel('Time (s)')
%     legend('DNN','RDNN')
%     grid on

e_rms_DNN=norm(rms(e_DNN'));
ftilde_rms_DNN=norm(rms(ftilde_DNN'));
u_rms_DNN=norm(rms(u_list_DNN));


e_rms_RDNN=norm(rms(e_RDNN'));
ftilde_rms_RDNN=norm(rms(ftilde_RDNN'));
u_rms_RDNN=norm(rms(u_list_RDNN));



    Architecture=["DNN";"RDNN"];
    RMS_Tracking_Error=[e_rms_DNN;e_rms_RDNN];
    RMS_Approximation_Error=[ftilde_rms_DNN;ftilde_rms_RDNN];
    Control_Inputs=[u_rms_DNN;u_rms_RDNN];
    Errors=table(Architecture,RMS_Tracking_Error,RMS_Approximation_Error,Control_Inputs)

    
