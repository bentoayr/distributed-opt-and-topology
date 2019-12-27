%% test the spectrum of the different graphs
graph_type = 5;

evol_spectrum = [];
for numV = 5:5:2*80

    [~,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line ] = generate_graph_data(numV, graph_type);

    Walk_G = diag(sum(Adj_G).^(-1))*Adj_G;
    
    eig_Lap_G = eig(Lap_G);
    gamma_Lap_G = eig_Lap_G(2)/eig_Lap_G(end);
    w_Walk_G = abs(eig(full(Walk_G)));
    w_Walk_G = max(w_Walk_G(w_Walk_G < 0.9999999));

    Walk_line_G = diag(sum(Adj_line_G).^(-1))*Adj_line_G;
    eig_Lap_line_G = eig(Lap_line_G);
    gamma_Lap_line_G = eig_Lap_line_G(2)/eig_Lap_line_G(end);
    w_Walk_line_G = abs(eig(full(Walk_line_G)));
    w_Walk_line_G = max(w_Walk_line_G(w_Walk_line_G < 0.9999999));
    
    evol_spectrum = [evol_spectrum; numV*[gamma_Lap_G,gamma_Lap_line_G, 1-w_Walk_G, 1-w_Walk_line_G]];

end