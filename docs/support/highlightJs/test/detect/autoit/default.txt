#NoTrayIcon
#AutoIt3Wrapper_Run_Tidy=Y
#include <Misc.au3>

_Singleton(@ScriptName) ; Allow only one instance
example(0, 10)

Func example($min, $max)
	For $i = $min To $max
		If Mod($i, 2) == 0 Then
			MsgBox(64, "Message", $i & ' is even number!')
		Else
			MsgBox(64, "Message", $i & ' is odd number!')
		EndIf
	Next
EndFunc   ;==>example