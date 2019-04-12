#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*\[Alpha]1[\[Delta]_,\[Theta]_]:=(((1-\[Delta])^2)/4)*(\[Theta])^2;*)
(*\[Alpha]2[\[Delta]_,\[Theta]_]:=(((1+\[Delta])^2)/4)* (\[Theta])^2;*)
(*\[Gamma]1[\[Delta]_,\[Theta]_]:=((1-\[Delta])/2)*\[Theta]+Sqrt[\[Alpha]1[\[Delta],\[Theta]]+\[Delta]];*)
(*\[Gamma]2[\[Delta]_,\[Theta]_]:=((1+\[Delta])/2)*\[Theta]+Sqrt[\[Alpha]2[\[Delta],\[Theta]]-\[Delta]];*)
(*\[Gamma]3[\[Delta]_,\[Theta]_]:=Sqrt[\[Delta]];*)
(**)
(*b[\[Theta]_]:=(-2+\[Theta]^2+2 Sqrt[1-\[Theta]^2])/\[Theta]^2;*)
(*\[Gamma][\[Delta]_,\[Theta]_]:=\[Gamma]1[Abs[\[Delta]],\[Theta]]*If[\[Delta]>=0&&\[Delta]<1,1,0]+\[Gamma]2[Abs[\[Delta]],\[Theta]]*If[\[Delta]>=b[\[Theta]]&&\[Delta]<0,1,0]+\[Gamma]3[Abs[\[Delta]],\[Theta]]*If[\[Delta]<b[\[Theta]]&&\[Delta]>=-1,1,0];*)


(* ::Input:: *)
(*FullSimplify[Series[\[Gamma]1[\[Delta],\[Theta]],{\[Theta],1,3}],Assumptions->\[Delta]<1&&\[Delta]>0]*)
(*FullSimplify[Series[\[Gamma]2[\[Delta],\[Theta]],{\[Theta],1,3}],Assumptions->\[Delta]<1&&\[Delta]>0]*)


(* ::Input:: *)
(*FullSimplify[((C-2)+Sqrt[(C-2)^2-C^2])/C/.C->(\[Lambda])^2,Assumptions->\[Lambda]<1&&\[Lambda]>-1]*)


(* ::Input:: *)
(*Plot[{-1,(-2+\[Lambda]^2+2 Sqrt[1-\[Lambda]^2])/\[Lambda]^2},{\[Lambda],-1,1}]*)


Manipulate[Plot[{\[Gamma][\[Delta],\[Theta]],\[Gamma]1[Abs[\[Delta]],\[Theta]],\[Gamma]2[Abs[\[Delta]],\[Theta]],\[Gamma]3[Abs[\[Delta]],\[Theta]],\[Gamma]3[Abs[b[\[Theta]]],\[Theta]]},{\[Delta],-1,1},PlotRange->{0,1}],{\[Theta],0,1}]


\[Gamma]2[-\[Delta],\[Theta]]//FullSimplify//Series[#,{\[Theta],1,3}]&


Series[-b[\[Theta]],{\[Theta],1,3}]


\[Gamma]3[-b[\[Theta]],\[Theta]]//FullSimplify


Series[Sqrt[1-x],{x,0,3}]


b[0.4]


1/2 (1-\[Delta]+Sqrt[(1+\[Delta])^2])+(1/2-\[Delta]/2+(-1+\[Delta])^2/(2 Sqrt[(1+\[Delta])^2])) (\[Theta]-1)/.\[Delta]->b[\[Theta]]//FullSimplify//Series[#,{\[Theta],1,3}]&//FullSimplify


\[Gamma]2[-b[0.99],0.99]
\[Gamma]3[-b[0.99],0.99]


\[Gamma]3[-b[\[Theta]],\[Theta]]//FullSimplify
\[Gamma]2[-b[\[Theta]],\[Theta]]//FullSimplify


Series[(1-Sqrt[1-\[Theta]^2])/\[Theta],{\[Theta],1,3}]
