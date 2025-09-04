function [e,ftilde,u_list,vecV_list,x,f_list] = RDNN_func(k,L,s,thresh,r,act,L_in,L_out, L_vec,vecV,step_size,simtime,x,ke,ks,Gamma)

act = "tanh";

%thresh = 2;

time_length=simtime/step_size;
% n = [s+1 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4];
n = [s+1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3];

R = cell(1,k);
R{1} = eye(s,s);

for i = 2:k+1
    R{i} = eye(L,L);
end

% e=[0;0];

counter = 0;
%r = 10;

for i=1:time_length
    counter = counter + 1;
    t=(i-1)*step_size;       %Time
    

    if t < thresh
        if counter > r
            for j = 2:k+1
                positions = randperm(L);
                positions = positions(1:n(j));
                for m = 1:n(j)
                    R{j}(positions(m),positions(m)) = 1;
                end
            end
    
        end
    else 
        for j = 2:k+1
            R{j} = eye(L,L);%/(L/n(j));
        end
        Gamma = 20;
    end
   

    xi=x(:,i);               %State
    xdi=[1;1;1;1;1]+[sin(2*t)+cos(0.5*t)^2;-cos(t);sin(3*t)+cos(-2*t); sin(t)^2; -sin(2*t)^2*cos(t)];  %Desired Trajectory
    xdi_dot=[2*cos(2*t)-cos(0.5*t)*sin(0.5*t);sin(t);3*cos(3*t)-2*sin(2*t); 2*sin(t)*cos(t); sin(2*t)^2*sin(t)-4*sin(2*t)*cos(2*t)];
    ei=xi-xdi;
    e(:,i)=ei;
    
    
    %DNN
    xda=[xi];  %Augmented with 1 for Bias
    
    
 
    [Lambdas,Phi,Phi_prime]=blockgrads_RDNN("tanh",R, vecV,xda,k,L,L_in,L_out);
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
    
    
    
    if counter > r & t < thresh
        counter = 0;
        for j = 2:k+1
            R{j} = zeros(L,L);
        end
    end



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

