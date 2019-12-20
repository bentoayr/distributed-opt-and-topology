function [evol, evol_X] = PDMM_non_conv(p,q,X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, D, ProxF , compute_objective, E1, E2, delta, target)

    dim = size(X_init,1);
    numV = size(X_init,2);
    
    X = X_init;
    U = U_init;
    
    evol = nan(num_iter,1);
    evol_X = nan(dim,numV,100);

    for t = 1:num_iter
        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            
            N = mean(  X_old(:,:,Neig_e,1) - U_old(:,:,Neig_e,e) , 3);
            % because of the way the PO was coded, we need to correct the
            % value of N.
            N = (rho*numE*length(Neig_e)*N + delta*target/numV)/(rho*numE*length(Neig_e) + delta/numV);
            
            % because of the way the PO was coded, we need to correct the
            % value of rho
            X(:,:,e,1) =   ProxF( p,q,  N    , rho*numE*length(Neig_e) + delta/numV , D, e ,E1, E2  );     
        end
        
        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            U(:,:,e, Neig_e) = U(:,:,e, Neig_e) + alp*(- U(:,:,e, Neig_e) + permute( -U_old(:,:,Neig_e,e) - repmat(X(:,:,e,1),1,1,length(Neig_e),1) +  X_old(:, :,Neig_e,1 ), [1, 2, 4, 3]));
        end
        
        AveX = mean(X(:,:,:,1),3);
        
        err =  compute_objective(AveX,D,p,q,E1,E2,delta,target);
        
        evol(t) = err;
        
        if (num_iter - t < 100)
           evol_X( : , : , 100 - (num_iter - t) ) = AveX; 
        end
        
        %evol = [evol, err ];
        %plot([1:1:t*1],evol'); 
        %scatter(X(1,:,1,1)',X(2,:,1,1)'); % we can vizualize the position of the points
        %drawnow;
    end
    
end