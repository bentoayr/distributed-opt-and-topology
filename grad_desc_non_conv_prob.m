function [evol, evol_X] = grad_desc_non_conv_prob(X_init, alf, compute_objective,GradF,num_iter, Adj_G, D, numE ,E1,E2, delta, target)

    dim = size(X_init,1);
    numV = size(X_init,2);
    
    evol = nan(num_iter,1);

    evol_X = nan(dim,numV,100);
    

    X = X_init;

    for t = 1:num_iter
        
        for i = 1:numV
            X(:,i) = X(:,i) - alf*GradF(X, i , Adj_G, D,numE, delta,target,numV);
        end        
        err = compute_objective(X,D,2,2,E1,E2,delta,target);
        
        evol(t) = err;
        
        if (num_iter - t < 100)
           evol_X(:,:,100 - (num_iter - t)) = X; 
        end
    
        %scatter(X(1,:)',X(2,:)')
        %drawnow;
        
    end
    
    
end

    
 
