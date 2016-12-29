(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["SomePkg`"]

Begin["`Private`"]

SomeFn[ns_List] := Fold[Function[{x, y}, x + y], 0, Map[# * 2 &, ns]];
Print[$ActivationKey];

End[] (* End Private Context *)

EndPackage[]
