function [evol, evol_Z] = ADMM_over_relaxed_node_Z_node_non_conv(p,q,X_init, U_init, Z_init, rho, gamma, numE, num_iter, num_iter_last_hist, ProxFPair ,compute_objective,Adj_G, D, E1, E2, delta, target)

    dim = size(Z_init,1);
    numV = size(Z_init,2);

    X = X_init;
    Z = Z_init;
    U = U_init;

    evol = nan(num_iter,1);
    evol_Z = nan(dim,numV,num_iter_last_hist);

    for t = 1:num_iter

        for e = 1:numE
            
            i = E1(e); j = E2(e);
            invdegi = 1/length(find(Adj_G(i,:)));
            invdegj = 1/length(find(Adj_G(j,:)));
            
            % because of the way the PO was coded, we need to correct the
            % value of rho1 and rho2
            rho1 = rho + (delta*invdegi)/numV;
            rho2 = rho + (delta*invdegj)/numV;
            
            N1 = Z(:,i) - U(:,1,e);
            N2 = Z(:,j) - U(:,2,e);
            
            % because of the way the PO was coded, we need to correct the
            % value of N1 and N2
            N1 = (rho*N1 + target*((delta*invdegi)/numV))/rho1;
            N2 = (rho*N2 + target*((delta*invdegj)/numV))/rho2;
            
            
            [X1,X2] =  ProxFPair(p,q, N1 , N2  , rho1*numE, rho2*numE , D(i,j) );                
            X(:,1,e) = X1; X(:,2,e) = X2;
        end
        
        Z_old = Z;
        for i = 1:numV
            e1Neigh = find(E1 == i);
            e2Neigh = find(E2 == i);            
            Z(:,i) = (1-gamma)*Z(:,i) + (sum(gamma*X(:,1,e1Neigh) + U(:,1,e1Neigh),3) + sum(gamma*X(:,2,e2Neigh) + U(:,2,e2Neigh),3)) / (length(e1Neigh) + length(e2Neigh));
        end
        
        U_old = U;
        for e = 1:numE
            i = E1(e); j = E2(e);
            U(:,1, e) = U_old(:,1, e) +  gamma*X(:,1,e)  - Z(:,i)  + (1-gamma)*Z_old(:,i); 
            U(:,2, e) = U_old(:,2, e) +  gamma*X(:,2,e)  - Z(:,j)  + (1-gamma)*Z_old(:,j); 
        end
        
        err =  compute_objective(Z,D,p,q,E1,E2,delta, target);
                
        evol(t) = err;
        
        if (num_iter - t < num_iter_last_hist)
           evol_Z( : , : , num_iter_last_hist - (num_iter - t) ) = Z; 
        end
        
        
        %evol = [evol, err ];
        %plot([1:1:t*1],evol'); 
        %scatter(Z(1,:)',Z(2,:)'); % we can vizualize the position of the points
        %drawnow;
    end

end