function [out,left_limit_new,right_limit_new,curr_parameter] = do_bisection_search(left_limit, right_limit, curr_parameter, num_steps, val_hist, curr_val)

    % while we have not scanned all of the points to test, move to the next one
    if (curr_parameter < right_limit)
        out = [val_hist, curr_val];
        curr_parameter = curr_parameter + (right_limit - left_limit)/num_steps;
        left_limit_new = left_limit;
        right_limit_new = right_limit;
        
    else
        [ ~ , ix ] = min(val_hist);
        left_limit_new = left_limit + (ix-2)*(right_limit - left_limit)/num_steps;
        right_limit_new = left_limit + (ix)*(right_limit - left_limit)/num_steps;
        curr_parameter = left_limit_new;
        out = [];
    end

end