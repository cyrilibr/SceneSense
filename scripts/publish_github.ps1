# Executer dans PowerShell apres l'installation de Git et GitHub CLI.
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'Installer GitHub CLI sur https://cli.github.com/ puis relancer.'
}
gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Se connecter avec gh auth login puis relancer.' }
git remote get-url origin 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { throw 'Un remote origin existe deja. Publication annulee.' }
gh repo create SceneSense-A4 --private --source=. --remote=origin --push
if ($LASTEXITCODE -ne 0) { throw 'La publication GitHub a echoue.' }
gh repo view --json url --jq .url
