# Path to your projects folder
$BaseDir = "D:\Food-Quality-Detector-main"

# Tracker file to remember last uploaded index
$Tracker = Join-Path $BaseDir ".upload_tracker"

# Create tracker if missing
if (!(Test-Path $Tracker)) {
    Set-Content -Path $Tracker -Value "0"
}

# Read current index
$LastIndex = [int](Get-Content $Tracker)

# Get ALL files inside all project folders (sorted)
$AllFiles = Get-ChildItem -Recurse -File -Path $BaseDir |
           Sort-Object FullName |
           Select-Object -ExpandProperty FullName

$TotalFiles = $AllFiles.Count

# Batch size
$BatchSize = 5

# End index for this run
$EndIndex = $LastIndex + $BatchSize
if ($EndIndex -gt $TotalFiles) {
    $EndIndex = $TotalFiles
}

# Select next 5 files
$FilesToCommit = $AllFiles[$LastIndex..($EndIndex-1)]

# Stop if no more files left
if ($FilesToCommit.Count -eq 0) {
    Write-Output "No files left to commit."
    exit
}

# Go to project root Git repository
Set-Location $BaseDir

# Add files to git
foreach ($File in $FilesToCommit) {
    git add "$File"
}

# Commit with timestamp
$Time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Daily upload: Files $($LastIndex+1) to $EndIndex on $Time"

# Push to GitHub
git push origin main

# Update tracker
Set-Content -Path $Tracker -Value $EndIndex
