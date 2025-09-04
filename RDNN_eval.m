function [F_error_rms]=RDNN_eval(k,L,L_in,L_out,vecV,X_rand, R, xda)

N=length(X_rand(1,1,:));
for i=1:N
    Xi=X_rand(:,:,i);
    X(:,i)=Xi;
     [Lambdas,Phi,Phi_prime] = blockgrads_RDNN('tanh',R,vecV,xda,k,L,L_in,L_out);
     f=[Xi(1)*Xi(2)^2*tanh(Xi(2)) + sin(Xi(1))^2;...
        cos(Xi(1)+Xi(2)+Xi(3))^3-exp(Xi(2))^2+Xi(1)*Xi(2);...
        Xi(3)^2*log(1+abs(Xi(1)-Xi(2))); ...
        sin(Xi(1)+Xi(2)^2)^2-exp(Xi(3))^0.5+(Xi(1)+Xi(3))*Xi(2);...
        tanh(Xi(2)) + cos(Xi(1))^2;];
    flist(:,i)=f;
    Philist(:,i)=Phi;
    F_error(:,i)=f-Phi;
end

F_error_rms=rms(vecnorm(F_error));