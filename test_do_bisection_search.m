left_limit = -1;
right_limit = 1;
num_steps = 4;
curr_parameter = left_limit;
val_hist = [];

while(right_limit - left_limit > 0.01)

    curr_val = curr_parameter^2;
    
    [val_hist,left_limit,right_limit,curr_parameter] = do_bisection_search(left_limit, right_limit, curr_parameter, num_steps, val_hist, curr_val);
    
end