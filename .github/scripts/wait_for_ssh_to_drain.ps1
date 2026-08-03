function Get-SSH-Users {
    # Gets ssh sessions for all users not named SYSTEM
    Get-CimInstance -ClassName Win32_Process -Filter "Name = 'sshd.exe'" |
        Get-CimAssociatedInstance -Association Win32_SessionProcess |
        Get-CimAssociatedInstance -Association Win32_LoggedOnUser |
        Where-Object {$_.Name -ne 'SYSTEM'} |
        Measure-Object
}

$usersLoggedOn = Get-SSH-Users

Write-Output "Holding runner until all ssh sessions have logged out"
while ($usersLoggedOn.Count -gt 0) {
    $usersLoggedOn = Get-SSH-Users
    Write-Output "."
    Start-Sleep -s 5
}
