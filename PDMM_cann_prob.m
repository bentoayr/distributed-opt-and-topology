function evol = PDMM_cann_prob(X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G,ProxF ,  E1, E2, delta, target)

    X = X_init;
    
    %XAve = zeros(dim,numE,1);
    U = U_init;
    
    %U_old = U;
    
    evol = [];
    
    for t = 1:num_iter
        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            X(:,e,1) =   ProxF(  mean(  X_old(:,Neig_e,1) - U_old(:,Neig_e,e) , 2)   ,   e   , rho*length(Neig_e) , E1, E2, numE, delta, target);     
        end
        
        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            U(:,e, Neig_e) = U(:,e, Neig_e) + alp*(- U(:,e, Neig_e) + permute( -U_old(:,Neig_e,e) - repmat(X(:,e,1),1,length(Neig_e),1) +  X_old(:, Neig_e,1 ), [1, 3, 2]));
        end
        
        
        
        evol = [evol, log(norm( X    - target)) ];
        
        %XAve = XAve + X; % the paper asks to compute the average in time
        %evol = [evol, log(norm( X/t    - target)) ];
        
        %plot([1:1:t*1],evol'); 
        %drawnow;
    end



end