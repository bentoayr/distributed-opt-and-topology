function evol = ADMM_edge_edge_no_Z_cann_prob(X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, ProxF , numEline, E1line, E2line, E1, E2, delta, target)

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
            Neig_e_ix = (E1line == e | E2line == e);
            X(:,e) =   ProxF(  mean(  X_old(:,Neig_e) - U(:,Neig_e_ix).*(sign(Neig_e - e)) , 2)    ,   e   , rho*length(Neig_e) ,  E1, E2, numE, delta, target);
        end

        for linee = 1:numEline
            e1 = E1line(linee);
            e2 = E2line(linee);

            U(:,linee) = U_old(:,linee) + alp*( X(:,e1) - X(:,e2) );
        end

        evol = [evol, log(norm( X    - target)) ];        
    end


end