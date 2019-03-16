%% testing different algorithms for distributed optimization

n = 10;
G = rand(n) > 0.5;

[E1, E2] = find(G);

numE = length(E1);

X = rand(n,numE);
Y = rand(n,numE);
Theta = rand(n,numE);

for e = 1:length(numE)
   Theta = GradConjF(X(:,e),e);
    
end



% this function computes the gradient of the i-th function in the objective
% that we are trying to optimize
function [GRAD] = GradConjF(X, i)

 %

end