# Create Desktop Shortcut for Resume Analyzer
$desktopPath = [System.Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopPath "Resume Analyzer.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)

$shortcut.TargetPath = "C:\resume Analyzer\quick_start.bat"
$shortcut.WorkingDirectory = "C:\resume Analyzer"
$shortcut.Description = "Resume Analyzer - AI-Powered Resume Analysis"
$shortcut.IconLocation = "C:\Windows\System32\shell32.dll,13"

$shortcut.Save()

Write-Host "Desktop shortcut created successfully!"
Write-Host "You can now double-click 'Resume Analyzer' on your desktop to start the application"
