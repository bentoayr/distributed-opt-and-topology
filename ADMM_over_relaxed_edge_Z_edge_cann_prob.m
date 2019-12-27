function [evol, evol_X] = ADMM_over_relaxed_edge_Z_edge_cann_prob(X_init, U_init, Z_init, rho, gamma, numE, num_iter, num_iter_last_hist, Adj_line_G, ProxF ,  E1, E2, delta, target)

    X = X_init;

    numV = size(X,1);
    
    %XAve = zeros(dim,numE,1);
    U = U_init;
    Z = Z_init;
    
    %U_old = U;

    evol = nan(num_iter,1);
    evol_X = nan(numV,num_iter_last_hist);

    
    for t = 1:num_iter

        U_old = U;
        Z_old = Z;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            X(:,e,1) =   ProxF(  mean( -permute(U(:,e,Neig_e),[1,3,2])  + Z(:,Neig_e,e) , 2)    ,   e   , rho*length(Neig_e) ,E1, E2, numE, delta, target);
        end

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            Z(:,e, Neig_e) = (1- gamma)*Z_old(:,e, Neig_e) +  0.5*(  U_old(:,e, Neig_e)  + permute( gamma*X(:,Neig_e,1)  + U_old(:,Neig_e,e) + repmat(gamma*X(:,e,1),1,length(Neig_e),1) , [1, 3, 2]) );
            U(:,e, Neig_e) = U_old(:,e, Neig_e) - Z(:,e, Neig_e) + (1-gamma)*Z_old(:,e, Neig_e) + permute(repmat(gamma*X(:,e,1),1,length(Neig_e),1) , [1, 3, 2])    ;
        end
        
        
        AveX = mean(X(:,:,1),2);
        
        evol(t) = log(norm( X    - target));
        
         if (num_iter - t < num_iter_last_hist)
           evol_X( : , num_iter_last_hist - (num_iter - t) ) = AveX; 
        end
        

    end




end