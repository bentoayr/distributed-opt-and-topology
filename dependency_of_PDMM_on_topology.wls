#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*\[Alpha]1[\[Delta]_,\[Theta]_]:=(1-\[Delta])^2/4 (\[Theta])^2;*)
(*\[Alpha]2[\[Delta]_,\[Theta]_]:=(1+\[Delta])^2/4 (\[Theta])^2;*)
(*\[Gamma]1[\[Delta]_,\[Theta]_]:=((1-\[Delta])/2)*\[Theta]+Sqrt[\[Alpha]1[\[Delta],\[Theta]]+\[Delta]];*)
(*\[Gamma]2[\[Delta]_,\[Theta]_]:=((1+\[Delta])/2)*\[Theta]+Sqrt[\[Alpha]2[\[Delta],\[Theta]]-\[Delta]];*)
(*\[Gamma]3[\[Delta]_,\[Theta]_]:=Sqrt[\[Delta]];*)


(* ::Input:: *)
(*FullSimplify[Series[\[Gamma]1[\[Delta],\[Theta]],{\[Theta],1,3}],Assumptions->\[Delta]<1&&\[Delta]>0]*)
(*FullSimplify[Series[\[Gamma]2[\[Delta],\[Theta]],{\[Theta],1,3}],Assumptions->\[Delta]<1&&\[Delta]>0]*)


(* ::Input:: *)
(*FullSimplify[((C-2)+Sqrt[(C-2)^2-C^2])/C/.C->(1-\[Lambda])^2,Assumptions->\[Lambda]<1&&\[Lambda]>-1]*)


(* ::Input:: *)
(*Plot[{1,Abs[(-1+(-2+\[Lambda]) \[Lambda]+2 Sqrt[-(-2+\[Lambda]) \[Lambda]])/(-1+\[Lambda])^2]},{\[Lambda],-1,1}]*)
