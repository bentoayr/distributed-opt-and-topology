function [ evol , evol_X ] = ADMM_edge_edge_no_Z_cann_prob(X_init, U_init, rho, alp, numE, num_iter,  num_iter_last_hist, Adj_line_G, ProxF , numEline, E1line, E2line, E1, E2, delta, target)

    dim = size(X_init,1);

    X = X_init;

    %XAve = zeros(dim,numE,1);
    U = U_init;

    %U_old = U;

    evol = nan( num_iter , 1 );
    evol_X = nan( dim , numE , num_iter_last_hist );
    
    for t = 1:num_iter

        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            Neig_e_ix = (E1line == e | E2line == e);
            X(:,e) =   ProxF(  mean(  X_old(:,Neig_e) - U(:,Neig_e_ix).*(sign(Neig_e - e)) , 2)    ,   e   , rho*length(Neig_e) ,  E1, E2, numE, delta, target);
        end

        for linee = 1:numEline
            e1 = E1line(linee);
            e2 = E2line(linee);

            U(:,linee) = U_old(:,linee) + alp*( X(:,e1) - X(:,e2) );
        end

        evol(t) = log(norm( X    -  target,'fro')) ;  
        if (num_iter - t < num_iter_last_hist)
           evol_X( :, : , num_iter_last_hist - (num_iter - t) ) = X; 
        end
    end


end