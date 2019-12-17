function evol = grad_desc_cann_prob(X_init, alf, num_iter, Lap_G, numE , delta, target)

    evol = [];
    X = X_init;
    for t = 1:num_iter
        X = X - alf*(Lap_G*X/(numE) + delta*(X-target));

        evol = [evol, log(norm( X - target ,1)) ];

    end


end