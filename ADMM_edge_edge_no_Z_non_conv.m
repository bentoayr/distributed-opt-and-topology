function [evol, evol_X] = ADMM_edge_edge_no_Z_non_conv(p,q,X_init, U_init, rho, alp, numE, numEline, num_iter,  Adj_line_G, D, ProxF , compute_objective, E1, E2,E1line,E2line, delta, target)

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
            Neig_e_ix = find(E1line == e | E2line == e);
            S = zeros(1,1,length(Neig_e_ix));
            S(1,1,:) = sign(Neig_e - e);
            
            N = mean(  X_old(:,:,Neig_e) - U(:,:,Neig_e_ix).*repmat(S,dim,numV,1) , 3);
            % because of the way the PO was coded, we need to correct the
            % value of N
            N = (rho*numE*length(Neig_e)*N + delta*target/numV)/(rho*numE*length(Neig_e) + delta/numV);
            
            % because of the way the PO was coded, we need to correct the
            % value of rho
            X(:,:,e) =  ProxF( p,q, N ,  rho*numE*length(Neig_e) + delta/numV  ,  D, e ,E1, E2  );     

        end
        
        for linee = 1:numEline
            e1 = E1line(linee);
            e2 = E2line(linee);
            
            U(:,:,linee) = U_old(:,:,linee) + alp*( X(:,:,e1) - X(:,:,e2) );
        end
        
        AveX = mean(X(:,:,:),3);
        
        err =  compute_objective(AveX,D,p,q,E1,E2,delta,target);
        
        evol(t) = err;
        
        if (num_iter - t < 100)
           evol_X( : , : , 100 - (num_iter - t) ) = AveX; 
        end
        
        %err =  log(compute_objective(X(:,:,1),D,p,q,E1,E2));
        %evol = [evol, err ];
        %plot([1:1:t*1],evol');
        
        %scatter(X(1,:,1)',X(2,:,1)'); % we can vizualize the position of the points
        %drawnow;
    end
    
end