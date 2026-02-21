' ArtificialBrain.vbs — Silent Windows launcher (double-click from Desktop)
' Uses the pre-built standalone binary — no Python startup overhead.
' Copy this file to your Windows Desktop.

Dim WshShell
Set WshShell = CreateObject("WScript.Shell")

' Run the pre-built Linux binary through WSL2 — no terminal window appears
WshShell.Run "wsl.exe -d kali-linux -- /home/rabindra18/neuralNetwork/dist/ArtificialBrain/ArtificialBrain", 0, False

Set WshShell = Nothing
