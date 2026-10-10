<#
.SYNOPSIS
    Prepares and stages the RecStage backend for deployment to Hugging Face Spaces.

.DESCRIPTION
    Clones the Hugging Face Space repository to a temporary directory, copies
    only the runtime backend code and artifacts, writes the Hugging Face Spaces
    metadata README.md, commits using the GitHub no-reply email, and displays
    the exact git push command.

.PARAMETER SpaceRepoUrl
    The HTTPS or SSH clone URL of the Hugging Face Space repository.
    Example: https://huggingface.co/spaces/username/recstage-api
#>

param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$SpaceRepoUrl
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Write-Host "=== RecStage Hugging Face Spaces Deployment Helper ===" -ForegroundColor Cyan
Write-Host "Source project: $ProjectRoot"
Write-Host "Target Space:   $SpaceRepoUrl"
Write-Host ""

# 1. Create a clean temporary directory
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("recstage_hf_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
Write-Host "[1/6] Created temporary directory: $TempDir" -ForegroundColor Green

try {
    # 2. Clone the Hugging Face Space repository
    Write-Host "[2/6] Cloning Space repository..." -ForegroundColor Green
    git clone $SpaceRepoUrl $TempDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone repository: $SpaceRepoUrl"
    }

    # 3. Copy runtime assets
    Write-Host "[3/6] Copying runtime files..." -ForegroundColor Green

    # Dockerfile
    Copy-Item (Join-Path $ProjectRoot "Dockerfile.hf") (Join-Path $TempDir "Dockerfile") -Force
    # requirements-deploy.txt
    Copy-Item (Join-Path $ProjectRoot "requirements-deploy.txt") (Join-Path $TempDir "requirements-deploy.txt") -Force

    # api/
    $TargetApi = Join-Path $TempDir "api"
    New-Item -ItemType Directory -Path $TargetApi -Force | Out-Null
    Copy-Item (Join-Path $ProjectRoot "api\*") $TargetApi -Recurse -Force -Exclude "__pycache__"

    # ml/ (without tests)
    $TargetMl = Join-Path $TempDir "ml"
    New-Item -ItemType Directory -Path $TargetMl -Force | Out-Null
    Get-ChildItem -Path (Join-Path $ProjectRoot "ml") -Exclude "tests", "__pycache__" | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $TargetMl $_.Name) -Recurse -Force
    }

    # artifacts/ (runtime only, skip images and caches)
    $TargetArtifacts = Join-Path $TempDir "artifacts"
    New-Item -ItemType Directory -Path $TargetArtifacts -Force | Out-Null
    $RuntimeArtifacts = @(
        "two_tower.pt",
        "faiss_index.bin",
        "faiss_index.npy",
        "item_embeddings.npy",
        "ranker.txt",
        "winning_users.json",
        "dataset_cache.pkl"
    )
    foreach ($art in $RuntimeArtifacts) {
        $artPath = Join-Path $ProjectRoot "artifacts\$art"
        if (Test-Path $artPath) {
            Copy-Item $artPath (Join-Path $TargetArtifacts $art) -Force
        }
    }

    # data/ files
    $TargetData = Join-Path $TempDir "data"
    New-Item -ItemType Directory -Path $TargetData -Force | Out-Null
    if (Test-Path (Join-Path $ProjectRoot "data\embedding_map.json")) {
        Copy-Item (Join-Path $ProjectRoot "data\embedding_map.json") (Join-Path $TargetData "embedding_map.json") -Force
    }
    if (Test-Path (Join-Path $ProjectRoot "data\posters.json")) {
        Copy-Item (Join-Path $ProjectRoot "data\posters.json") (Join-Path $TargetData "posters.json") -Force
    }

    # results/ metrics
    $TargetResults = Join-Path $TempDir "results"
    New-Item -ItemType Directory -Path $TargetResults -Force | Out-Null
    if (Test-Path (Join-Path $ProjectRoot "results\metrics.json")) {
        Copy-Item (Join-Path $ProjectRoot "results\metrics.json") (Join-Path $TargetResults "metrics.json") -Force
    }

    # 4. Write README.md with HF front matter
    Write-Host "[4/6] Writing Hugging Face Space README.md..." -ForegroundColor Green
    $HfReadmeContent = @"
---
title: RecStage API
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# RecStage Recommendation API

High-performance two-stage recommendation API combining PyTorch Two-Tower candidate retrieval with LightGBM LambdaMART re-ranking on MovieLens-1M. Built with FastAPI and containerized for Hugging Face Spaces.
"@
    Set-Content -Path (Join-Path $TempDir "README.md") -Value $HfReadmeContent -Encoding utf8

    # 5. Check file sizes and warn if > 10 MB
    Write-Host "[5/6] Checking staged file sizes..." -ForegroundColor Green
    $Files = Get-ChildItem -Path $TempDir -Recurse -File | Where-Object { $_.FullName -notmatch "\\\.git\\" }
    $Over10MB = @()
    foreach ($f in $Files) {
        $RelPath = $f.FullName.Substring($TempDir.Length + 1)
        $SizeMB = [Math]::Round($f.Length / 1MB, 2)
        $SizeKB = [Math]::Round($f.Length / 1KB, 1)
        if ($f.Length -gt 10MB) {
            Write-Host "  [WARN > 10MB] $RelPath ($SizeMB MB)" -ForegroundColor Yellow
            $Over10MB += $RelPath
        } else {
            Write-Host "  $RelPath ($SizeMB MB / $SizeKB KB)" -ForegroundColor DarkGray
        }
    }

    if ($Over10MB.Count -gt 0) {
        Write-Host "`nWARNING: The following files exceed 10 MB and may require Git LFS on Hugging Face Spaces:" -ForegroundColor Yellow
        $Over10MB | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow }
    } else {
        Write-Host "All files are under 10 MB (safe for standard git push without Git LFS)." -ForegroundColor Green
    }

    # 6. Git commit
    Write-Host "[6/6] Committing staged files..." -ForegroundColor Green
    Push-Location $TempDir
    try {
        git config user.name "Rishabh Yadav"
        git config user.email "214611814+rishabhyadav03@users.noreply.github.com"
        git add .
        $Status = git status --porcelain
        if ($Status) {
            git commit -m "chore(deploy): stage RecStage backend for Hugging Face Spaces"
            Write-Host "Commit created successfully." -ForegroundColor Green
        } else {
            Write-Host "No changes to commit (repository already up to date)." -ForegroundColor Cyan
        }

        $CurrentBranch = (git branch --show-current).Trim()
        if (-not $CurrentBranch) { $CurrentBranch = "main" }

        Write-Host ""
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "Deployment package is ready! To deploy, run the following:" -ForegroundColor Cyan
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "cd `"$TempDir`"" -ForegroundColor White
        Write-Host "git push origin $CurrentBranch" -ForegroundColor Yellow
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "Note: git will prompt for your Hugging Face credentials." -ForegroundColor DarkGray
        Write-Host "(Username: your HF username | Password: HF Access Token with write permissions)" -ForegroundColor DarkGray
        Write-Host ""
    } finally {
        Pop-Location
    }
} catch {
    Write-Error "Deployment preparation failed: $_"
}
