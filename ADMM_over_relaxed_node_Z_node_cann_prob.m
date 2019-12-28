function [evol , evol_Z] = ADMM_over_relaxed_node_Z_node_cann_prob(X_init, U_init, Z_init, rho, gamma, dim, numE, num_iter, num_iter_last_hist, ProxFPair , E1, E2, delta, target)

    X = X_init;
    Z = Z_init;
    U = U_init;

    numV = dim;
    evol = nan(num_iter,1);
    evol_Z = nan(dim,num_iter_last_hist);

    
    for t = 1:num_iter

        for e = 1:numE
            i = E1(e); j = E2(e);
            [X(:,e)] =  ProxFPair( [Z(i) - U(1,e);Z(j) - U(2,e)]   ,   e   , rho*numE , E1,E2,numE,delta,target);
        end

        Z_old = Z;
        for i = 1:dim
            e1Neigh = find(E1 == i);
            e2Neigh = find(E2 == i);
            Z(i) = (1-gamma)*Z(i)    +    (sum(gamma*X(1,e1Neigh) + U(1,e1Neigh),2) + sum(gamma*X(2,e2Neigh) + U(2,e2Neigh),2)) / (length(e1Neigh) + length(e2Neigh));
        end

        U_old = U;
        for e = 1:numE
            i = E1(e); j = E2(e);
            U(1, e) = U_old(1, e) +  gamma*X(1,e)  - Z(i) + (1-gamma)*Z_old(i) ;
            U(2, e) = U_old(2, e) +  gamma*X(2,e)  - Z(j) + (1-gamma)*Z_old(j) ;
        end

        err = log(norm( Z    - target));

        evol(t) = err ;
        if (num_iter - t < num_iter_last_hist)
           evol_Z( : , num_iter_last_hist - (num_iter - t) ) = Z; 
        end

    end



end