function rate = estimate_rate_out_of_plot(evol)


    mvm = movmean(diff(evol),length(evol)/10);
    mvst = movstd(diff(evol),length(evol)/10);
    [~, ix] = min( mvm + mvst );
    
    rate = mvm(ix);



end