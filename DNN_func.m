function [e,ftilde,u_list,vecV_list,x,f_list, Time_DNN, FLOPs_DNN] = DNN_func(k,L,s,act,L_in,L_out, L_vec,vecV,step_size,simtime,x,ke,ks,Gamma)
% e=[0;0];
time_length=simtime/step_size;
Time_DNN = zeros(time_length,1);
for i=1:time_length

    t=(i-1)*step_size;       %Time
    xi=x(:,i);               %State
    xdi=[1;1;1;1;1]+[sin(2*t)+cos(0.5*t)^2;-cos(t);sin(3*t)+cos(-2*t); sin(t)^2; -sin(2*t)^2*cos(t)];  %Desired Trajectory
    xdi_dot=[2*cos(2*t)-cos(0.5*t)*sin(0.5*t);sin(t);3*cos(3*t)-2*sin(2*t); 2*sin(t)*cos(t); sin(2*t)^2*sin(t)-4*sin(2*t)*cos(2*t)];
    ei=xi-xdi;
    e(:,i)=ei;
    
    
    %DNN
    xda=[xi];  %Augmented with 1 for Bias
    
    
 
    [Lambdas,Phi,Phi_prime, Time_DNN(i), FLOPs_DNN]=blockgrads_DNN(act,vecV,xda,k,L,L_in,L_out);
    vecVdot=Gamma*Lambdas'*ei;
    
    vecV=vecV+step_size*vecVdot;
    
    u=xdi_dot-Phi-ke*ei-ks*sign(ei);
    u_list(:,i)=u;
    
    %f=[xi(1)*xi(2)*tanh(xi(2))+sech(xi(1))^2;sech(xi(1)+xi(2))^2-sech(xi(2))^2];
    f=[xi(1)*xi(2)^2*tanh(xi(2)) + sin(xi(1))^2;...
        cos(xi(1)+xi(2)+xi(3))^3-exp(xi(2))^2+xi(1)*xi(2);...
        xi(3)^2*log(1+abs(xi(1)-xi(2))); ...
        sin(xi(1)+xi(2)^2)^2-exp(xi(3))^0.5+(xi(1)+xi(3))*xi(2);...
        tanh(xi(2)) + cos(xi(1))^2;];
    f_list(:,i)=f;

    xdot=f+u;
    x(:,i+1)=xi+step_size*xdot;   
    ftilde(:,i)=f-Phi;
    Phiplot(:,i)=Phi;
    vecV_list(:,i+1)=vecV;  
end


e(:,i+1)=x(:,i+1)-xdi;
ftilde(:,i+1)=ftilde(:,i);
% Phiplot(:,i+1)=Phi;
% time=(0:time_length)*step_size;
% subplot(3,1,1)
% plot(time,e)
% subplot(3,1,2)
% plot(time,ftilde)
% subplot(3,1,3)
% plot(time,vecV_list)
% ylim([-22 22])

