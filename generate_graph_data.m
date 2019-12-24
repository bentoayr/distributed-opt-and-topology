function [numV,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line ] = generate_graph_data(numV, type)
    
    switch type

        case 1
        %% complete graph
        G = ones(numV) - eye(numV);

        case 2
        %% complete bipartite graph
        numV = 2*ceil(numV/2);
        G = ones(numV);
        G(1:(numV/2),1:(numV/2)) = 0;
        G((numV/2)+1:end,(numV/2)+1:end) = 0;
        G(1:numV+1:end) = 0; % no diagonal
        case 3
        %% cyclic chain
        G = zeros(numV);
        G(2:numV+1:end) = 1;
        G(1,end) = 1;
        G = G + G';

        case 4
        %% K-cube
        G = 0;
        for i = 1:floor(log2(numV))
            G = [[G, eye(length(G))];[eye(length(G)),G]];
        end
        numV = 2^floor(log2(numV));  % output the correct number of nodes
        case 5
        %% K-hop lattice
        K = ceil(log(numV));
        G = zeros(numV);
        G(2:numV+1:end) = 1;
        G(K+1:numV+1:end) = 1;
        for i = 1:K
            G(i,end-(K-i)) = 1;
        end
        G = G + G';

        case 6
        %% periodic grid
        p = ceil(sqrt(numV));
        G = zeros(p*p);
        for i = 1:p
            for j = 1:p
                ix = i + p*(j-1);
                ix_l = 1+mod(i+1-1,p) + p*(j-1);
                ix_ll = 1+mod(i-1-1,p) + p*(j-1);
                ix_r = i + p*( 1 + mod(j+1 - 1,p) -1);
                ix_rr = i + p*( 1 + mod(j-1 - 1,p) -1);
                G(ix,ix_l) = 1;
                G(ix,ix_ll) = 1;
                G(ix,ix_r) = 1;
                G(ix,ix_rr) = 1;
            end
        end
        numV = p*p; % output the correct number of nodes
        case 7
        %% Erd?s?Rényi
        rng(1); % we use the same seed to guarantee that the results are reproducible
        p = 3*log(numV)/numV; % make sure the graph is connected (with high prob)
        G = rand(numV) > p;
        G = (triu(G,1) + triu(G,1)');

    end
    %% extract stuff from graph
     
    G = graph(G);
    
    Adj_G = adjacency(G);
    Lap_G = laplacian(G);
    Inc_G = abs(incidence(G)); % it seems like Matlab orders edges based on the "lexicographical" indices. So edge (1,2) comes before edge (2,3). Also, there is no edge (2,1) in G. Also, it seems like the incidence matlab function puts signs on the elements, even for undirected graphs. that is why we use the abs() outside of it.
    Adj_line_G = Inc_G'*Inc_G - 2*eye(G.numedges); % the relation between the line graph and the incidence matrix is well known. see e.g. https://en.wikipedia.org/wiki/Incidence_matrix#Undirected_and_directed_graphs

    line_G = graph(Adj_line_G);
    Lap_line_G = laplacian(line_G);

    E1 = G.Edges.EndNodes(:,1);
    E2 = G.Edges.EndNodes(:,2);
    E1line = line_G.Edges.EndNodes(:,1);
    E2line = line_G.Edges.EndNodes(:,2);

    numE = G.numedges;
    numEline = line_G.numedges;

end


