clc; clear; close all;

step_size=0.01;
simtime=10;
time_length=simtime/step_size;
x=[12;1;15; -15; 12];
ke=45;
ks=0.005;
Gamma=40;  %Adaptation Gain
        

%DNN Parameters

s=5;   % Size of the State; 
k=25;   % Total Number of Hidden Layers 

L1=3;  % Width of Each Hidden Layer 
L2=10;

L_in = (s);
L_out= s; 

L1_vec= (L_out+L_in+(k-1)*L1)*L1;
L2_vec= (L_out+L_in+(k-1)*L2)*L2;

vecV1=10*rand(L1_vec,1);
vecV2=10*rand(L2_vec,1);

% droput frequency
deltat = 10;       

[e_RDNN1,ftilde_RDNN1,u_list_RDNN1,vecV_list_RDNN1,x_RDNN1,f_list_RDNN1]=...
    DNN_func(k,L1,s,"tanh",L_in,L_out, L1_vec,vecV1,step_size,simtime,x,ke,ks,Gamma);
[e_RDNN2,ftilde_RDNN2,u_list_RDNN2,vecV_list_RDNN2,x_RDNN2,f_list_RDNN2]=...
    RDNN_func(k,L2,s,10,deltat,"tanh",L_in,L_out, L2_vec,vecV2,step_size,simtime,x,ke,ks,Gamma);

e_rms_RDNN1=norm(rms(e_RDNN1'));
ftilde_rms_RDNN1=norm(rms(ftilde_RDNN1'));
u_rms_RDNN1=norm(rms(u_list_RDNN1'));

e_rms_RDNN2=norm(rms(e_RDNN2'));
ftilde_rms_RDNN2=norm(rms(ftilde_RDNN2'));
u_rms_RDNN2=norm(rms(u_list_RDNN2'));

cost_RDNN1=e_rms_RDNN1^2+0.01*u_rms_RDNN1^2;
cost_RDNN2=e_rms_RDNN2^2+0.01*u_rms_RDNN2^2;

vecV_opt_RDNN1=vecV1;
cost_min_RDNN1=cost_RDNN1;

vecV_opt_RDNN2=vecV1;
cost_min_RDNN2=cost_RDNN2;

cost_min_RDNN1
cost_min_RDNN2

time=(0:(time_length))*step_size;


figure(1)

    
    subplot(2,1,1)
    plot(time,vecnorm(e_RDNN1),time,vecnorm(e_RDNN2),'LineWidth',1.5)
    ylabel('$||e||$','Fontsize',16,'Interpreter','latex', 'FontName','Times New Roman')
    legend('Narrow DNN','Lb-DDNN','Interpreter','latex','Fontsize',16,'Location','NorthEast','Orientation','Horizontal')
        
    set(gca, 'YLim', [-0.5 34.5], 'YTick', 0:5:30,...
     'YTickLabel', 0:5:30);
    set(gca, 'XLim', [-0.2 10.2], 'XTick', 0:1:10,...
     'XTickLabel', 0:1:10);
    
    subplot(2,1,2)
    plot(time,vecnorm(ftilde_RDNN1),time,vecnorm(ftilde_RDNN2), 'LineWidth', 1.5)
    ylabel('$||f(x)-\widehat{\Phi}||$','Fontsize',16,'Interpreter','latex', 'FontName','Times New Roman')
    xlabel('Time (s)','Fontsize',16,'Interpreter','latex', 'FontName','Times New Roman')
    legend('Narrow DNN','Lb-DDNN','Interpreter','latex','Fontsize',16, 'FontName','Times New Roman','Location','NorthEast','Orientation','Horizontal')
    
    set(gca, 'YLim', [-10 2000], 'YTick', 0:200:1900,...
     'YTickLabel', 0:200:1900);
    set(gca, 'XLim', [-0.2 10.2], 'XTick', 0:1:10,...
     'XTickLabel', 0:1:10);

    
% figure(2)
% 
%     plot(time,vecV_list_DNN)
%     ylabel('DNN Weight Estimates')
%     xlabel('Time (s)')
% %     ylim([-2 2])
%     grid on
% 
% figure(3)
% 
%     plot(time,vecV_list_RDNN1)
%     ylabel('RDNN Weight Estimates')
%     xlabel('Time (s)')
%     %ylim([-1 1])
%     grid on

% figure(4)
% 
%     plot(time,vecnorm(u_list_DNN),time,vecnorm(u_list_RDNN))
%     ylabel('Normalized Control Inputs')
%     xlabel('Time (s)')
%     legend('DNN','RDNN')
%     grid on

e_rms_RDNN1=norm(rms(e_RDNN1'));
ftilde_rms_RDNN1=norm(rms(ftilde_RDNN1'));
u_rms_RDNN1=norm(rms(u_list_RDNN1'));


e_rms_RDNN2=norm(rms(e_RDNN2'));
ftilde_rms_RDNN2=norm(rms(ftilde_RDNN2'));
u_rms_RDNN2=norm(rms(u_list_RDNN2'));

  

    Architecture=["RDNN1";"RDNN2"];
    RMS_Tracking_Error=[e_rms_RDNN1;e_rms_RDNN2];
    RMS_Approximation_Error=[ftilde_rms_RDNN1;ftilde_rms_RDNN2];
    Control_Inputs=[u_rms_RDNN1;u_rms_RDNN2];
    Errors=table(Architecture,RMS_Tracking_Error,RMS_Approximation_Error,Control_Inputs)
