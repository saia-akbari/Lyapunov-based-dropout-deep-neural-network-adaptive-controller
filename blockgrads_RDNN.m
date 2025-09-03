function [Lambdas,Phi,Phi_prime, Time_RDNN, Total_FLOPs] = blockgrads(act,R,vecV,eta,k,L,L_in,L_out) 

tic
Total_FLOPs = 0;
if(act=="tanh")

    V0=unvec(vecV(1:L_in*L),L,L_in);  %Input Layer Weight
    Total_FLOPs = Total_FLOPs + 2*L*L_in*L_in;
    R0V0 = V0*R{1};
    Total_FLOPs = Total_FLOPs + 2*L*L_in;
    phi=tanh(R0V0*eta);                 %First Layer Activation %phi_1
     %phi(1)=1;                         %Bias 
    
    
    phi_list=zeros(length(phi),k);
    phi_list(:,1)=phi;                %Need to create a list of activation at each layer of the DNN. Require to store them in memory for subsequent computation
    Total_FLOPs = Total_FLOPs + L;
    phidash=sech2(V0*eta);            %Activation Derivative/Gradient
     %phidash(1,1)=0;                   %Bias Accomodation 
    
    
    phidash_list=zeros(length(phi),k); % Need a list to store activation gradients in the memory.
    phidash_list(:,1)=phidash;    
    
    for j=1:k-1
        
        
        Vj=unvec(vecV(L_in*L+1+(j-1)*L*L:L_in*L+j*L*L),L,L); %V_j        
        Total_FLOPs = Total_FLOPs + 2*L*L*L;
        RjVj = Vj*R{j+1};
        Total_FLOPs = Total_FLOPs + 2*L*L;
        Total_FLOPs = Total_FLOPs + L;
        phi=tanh(RjVj*phi);          %phi_(j+1)  
        %phi(1)=1;                         %Bias 
        phi_list(:,j+1)=phi;
        Total_FLOPs = Total_FLOPs + L;
        phidash=sech2(RjVj*phi);
        %phidash(1,1)=0;                   %Bias Accomodation 
        phidash_list(:,j+1)=phidash;        
    end
        
    Vk=unvec(vecV(L_in*L+(k-1)*L*L+1:L_in*L+(k-1)*L*L+L_out*L),L_out,L);
    Total_FLOPs = Total_FLOPs + 2*L_out*L*L;
    RkVk = Vk*R{k+1};
    Total_FLOPs = Total_FLOPs + 2*L_out*L;
    Phi=RkVk*phi; 
    Total_FLOPs = Total_FLOPs + L*L_out*L;
 
    Lambdak=(kron(phi'*R{k+1},eye(L_out)));
    Lambdas(1:L_out,L_in*L+(k-1)*L*L+1:L_in*L+(k-1)*L*L+L_out*L)=Lambdak;
    Total_FLOPs = Total_FLOPs + 2*L_out*L;
    Xi=RkVk*diag(phidash);

    
    for j=flip(1:k-1)
        
        phi_j=phi_list(:,j);
        Total_FLOPs = Total_FLOPs + L*L*L + 2*L_out*L*L;
        Lambda=Xi*kron(phi_j'*R{j+1},eye(L)); 
        Lambdas(1:L_out,L_in*L+1+(j-1)*L*L:L_in*L+j*L*L)=Lambda;
        
        Vj=unvec(vecV(L_in*L+1+(j-1)*L*L:L_in*L+j*L*L),L,L); %V_j
        Total_FLOPs = Total_FLOPs + 2*L*L*L;
        RjVj = Vj*R{j+1};
        
        phidash_j=diag(phidash_list(:,j));
        Total_FLOPs = Total_FLOPs + 2*L_out*L*L + 2*L_out*L;
        Xi=Xi*RjVj*phidash_j;             
        
    end
    Total_FLOPs = Total_FLOPs + L_in*L*L_in;
    Total_FLOPs = Total_FLOPs + 2*L_out*L*L_in;
    Lambda0=Xi*kron(eta'*R{1}, eye(L));
    Lambdas(1:L_out,1:L_in*L)=Lambda0;
    Total_FLOPs = Total_FLOPs + 2*L_out*L*L_in;
    Phi_prime=Xi*R0V0;
    
else % here on out: not edited yet
    if(act=="relu")
        
        V0=unvec(vecV(1:L_in*L),L,L_in);  %Input Layer Weight
        phi=relu(V0*eta);                 %First Layer Activation %phi_1
         phi(1)=1;                         %Bias 
        

        phi_list=zeros(length(phi),k);
        phi_list(:,1)=phi;                %Need to create a list of activation at each layer of the DNN. Require to store them in memory for subsequent computation
        R0V0 = V0*R{0};
        phidash=st(R0V0*eta);            %Activation Derivative/Gradient
         phidash(1,1)=0;                   %Bias Accomodation 

        phidash_list=zeros(length(phi),k); % Need a list to store activation gradients in the memory.
        phidash_list(:,1)=phidash;    

        for j=1:k-1
            Vj=unvec(vecV(L_in*L+1+(j-1)*L*L:L_in*L+j*L*L),L,L); %V_j
            
            phi=relu(Vj*phi);          %phi_(j+1)               
             phi(1)=1;                         %Bias 
            phi_list(:,j+1)=phi;
            
            phidash=st(Vj*phi);
             phidash(1,1)=0;                   %Bias Accomodation 
            
            phidash_list(:,j+1)=phidash;        
        end

        Vk=unvec(vecV(L_in*L+(k-1)*L*L+1:L_in*L+(k-1)*L*L+L_out*L),L_out,L);
        Phi=Vk*phi; 




        Lambdak=(kron(eye(L_out),phi'));
        Lambdas(1:L_out,L_in*L+(k-1)*L*L+1:L_in*L+(k-1)*L*L+L_out*L)=Lambdak;

        Xi=Vk*diag(phidash);


        for j=flip(1:k-1)

            phi_j=phi_list(:,j);
            Lambda=Xi*kron(eye(L),phi_j'); 
            Lambdas(1:L_out,L_in*L+1+(j-1)*L*L:L_in*L+j*L*L)=Lambda;

            phidash_j=diag(phidash_list(:,j));
            Xi=Xi*Vj*phidash_j;             

        end

        Lambda0=Xi*kron(eta'*R{1},eye(L));
        Lambdas(1:L_out,1:L_in*L)=Lambda0;
        Phi_prime=Xi*V0;
    else
        blockgrads("tanh",vecV,eta,k,L,L_in,L_out)
    end
    
end
Time_RDNN = toc;
end
    

    
    
    
    
    
    
    