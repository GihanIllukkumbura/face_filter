param(
    [string]$VenvPath = ".venv"
)

python -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Venv ready. Activate with: $VenvPath\Scripts\Activate.ps1"
