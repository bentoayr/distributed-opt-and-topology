function [evol , evol_X]  = grad_desc_cann_prob(X_init, alf, num_iter,num_iter_last_hist, Lap_G, numE , delta, target)

    X = X_init;
    dim = size(X,1);

    evol = nan(num_iter,1);
    evol_X = nan(dim,num_iter_last_hist);
    
    for t = 1:num_iter
        X = X - alf*(Lap_G*X/(numE) + delta*(X-target));

        evol(t) = log(norm( X - target ,'fro')) ;

         if (num_iter - t < num_iter_last_hist)
           evol_X( : , num_iter_last_hist - (num_iter - t) ) = X; 
        end
        
    end

end