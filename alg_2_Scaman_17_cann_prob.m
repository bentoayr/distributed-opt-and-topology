function [evol,K,evol_Theta] = alg_2_Scaman_17_cann_prob(Y_init, Theta_init , GradConjF, AccGoss, num_iter,num_iter_last_hist, numE , Lap_line_G, delta, E1,E2,target,alpha, beta)

    dim = size(Theta_init,1);

    W = Lap_line_G;

    % the choice of the following values is according to Scaman et al. 2017, "Optimal algorithms for smooth and strongly convexdistributed optimization in networks"
    kappa_l = beta / alpha; 
    specW = sort(eig(W)); 
    gamma = specW(2)/specW(end);
    c1 = (1 - sqrt(gamma))/ (1 + sqrt(gamma));
    c2 = (1 + gamma) / (1 - gamma);
    c3 = 2/ ((1+gamma)*specW(end));
    K = floor(1 / sqrt(gamma));
    eta_2 = alpha*(1 + c1^(2*K))/((1 + c1^K)^2);
    mu_2 = ((1 + c1^K)*sqrt(kappa_l) - 1 + c1^K) / ((1 + c1^K)*sqrt(kappa_l) + 1 - c1^K);

    Y = Y_init*real(sqrtm(full(W)));
    X = Y;
    Theta = repmat(Theta_init,1,numE);

    evol = nan(num_iter,1);
    evol_Theta = nan(dim,numE,num_iter_last_hist);
    
    for t = 1:num_iter
        for e = 1:numE
            Theta(:,e) = GradConjF( X(:,e) , e ,delta, E1,E2,target);
        end
        Y_old = Y;
        Y = X - eta_2*AccGoss(X, W, K, c2, c3);
        X = (1 + mu_2)*Y - mu_2*Y_old;

        evol(t) = log(norm( Theta(:,:) - target ,'fro'));
        if (num_iter - t < num_iter_last_hist)
           evol_Theta( :,: , num_iter_last_hist - (num_iter - t) ) = Theta; 
        end
        
    end
end