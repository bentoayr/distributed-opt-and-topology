function rate = estimate_rate_out_of_plot(evol)

    % this is some heuristic to try to estimate the asymptotic rate of
    % convergence from the bookeping of the error
    mvm = movmean(diff(evol),length(evol)/10);
    mvst = movstd(diff(evol),length(evol)/10);
    [~, ix] = min( mvm + mvst ); % this is to capture the part of the curve that has a stable linear rate of convergence
    
    rate = mvm(ix);



end