#!/usr/bin/env wolframscript
(* ::Package:: *)

(*Exansion of our converge rate for omega* close to 1*)
Series[4/(3-Sqrt[(2-2Sqrt[1-(1-d)^2])/(2+2Sqrt[1-(1-d)^2])])-1,{d,0,1}]//TeXForm


(*Some numerical calculations on the spectrum of different graphs*)
list={};

For[i = 3,i<=6,i=i+1,{
For[j=1,j<100,j++,{
NumNodes = 50;
G=RandomGraph[DegreeGraphDistribution[Table[i,{NumNodes}]]];
AG=G//AdjacencyMatrix//Normal;

LG=KirchhoffMatrix[G];
SpecLG=Eigenvalues[LG+0.0];

WG=Inverse[(AG+LG)].AG;
SpecWG=Eigenvalues[WG+0.0];

lG=LineGraph[G];
AlG=lG//AdjacencyMatrix//Normal;
LlG=KirchhoffMatrix[lG];
SpecLlG=Eigenvalues[LlG+0.0];

ADMMTime =1/Sqrt[1-RankedMax[SpecWG,2]];
GDTime =1/(1-(RankedMax[SpecLG,1]-RankedMin[SpecLG,2])/(RankedMax[SpecLG,1]+RankedMin[SpecLG,2]));
FBTime =1/Sqrt[RankedMin[SpecLlG,2]/RankedMax[SpecLlG,1]];
AppendTo[list,{ADMMTime,GDTime,FBTime}];
}];
}];
Partition[list[[;;,;;]],99]//Transpose//Mean//Transpose//ListPlot[#,Joined->True,PlotRange->All,PlotMarkers->Automatic,,Frame->True,PlotLegends->{"ADMM","GD","FB"}]&


(* ::Input:: *)
(*Inverse[({*)
(* {1+d, -1},*)
(* {-1, 1+d}*)
(*})]//FullSimplify//MatrixForm*)
