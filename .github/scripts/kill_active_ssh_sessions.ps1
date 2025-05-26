function Get-SSH-Sessions {
    Get-Process sshd -IncludeUserName |
        Where-Object UserName -notLike "*SYSTEM*" |
        Select-Object Id
}

$runningSessions = Get-SSH-Sessions

foreach ($session in $runningSessions) {
    Stop-Process -id $session.Id
}
