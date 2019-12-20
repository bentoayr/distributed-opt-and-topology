function rate = estimate_rate_out_of_X_evol(X_evol)

    num_iter = size(X_evol,3);

    err_evol = sum(sum(abs(repmat(X_evol(:,:,end),1,1,num_iter) - X_evol),1),2);

    err_evol = permute(err_evol,[3,2,1]);
    
    rate = estimate_rate_out_of_plot(log(err_evol(1:end-(num_iter/10))));

end