function [evol , evol_X] = PDMM_cann_prob(X_init, U_init, rho, alp, numE, num_iter,num_iter_last_hist,  Adj_line_G,ProxF ,  E1, E2, delta, target)

    dim = size(X_init,1);

    X = X_init;
    
    %XAve = zeros(dim,numE,1);
    U = U_init;
    
    %U_old = U;
    
    evol = nan(num_iter,1);
    evol_X = nan(dim,numE,num_iter_last_hist);
    
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
        
        
        
        evol(t) = log(norm( X  - target,'fro')) ;
      
        if (num_iter - t < num_iter_last_hist)
           evol_X( : ,:, num_iter_last_hist - (num_iter - t) ) = X; 
        end
        
        %XAve = XAve + X; % the paper asks to compute the average in time
        %evol = [evol, log(norm( X/t    - target)) ];
        
        %plot([1:1:t*1],evol'); 
        %drawnow;
    end



end