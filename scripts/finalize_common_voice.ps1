$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$source = Join-Path $projectRoot 'voice\data\source\cv-corpus-27.0-2026-09-11\tr'
$clips = Join-Path $source 'clips'
$python = Join-Path $projectRoot 'voice\.venv\Scripts\python.exe'
$importer = Join-Path $projectRoot 'voice\scripts\import_common_voice_scripted.py'
$output = Join-Path $projectRoot 'voice\data\common_voice_scripted'
$log = Join-Path $projectRoot 'voice\artifacts\common-voice-finalize.log'
$taskName = 'TurkishCommonVoiceFinalize'
$expectedClips = 126794

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
function Write-Log([string]$message) {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format s) $message"
}

if (Get-Process -Name tar -ErrorAction SilentlyContinue) {
    Write-Log 'Waiting: archive extraction is still running.'
    exit 0
}

if (-not (Test-Path -LiteralPath $clips)) {
    Write-Log 'Stopped: clips directory is missing.'
    exit 1
}

$clipCount = (Get-ChildItem -LiteralPath $clips -File -Filter '*.mp3' | Measure-Object).Count
if ($clipCount -lt $expectedClips) {
    Write-Log "Stopped: extraction ended early ($clipCount/$expectedClips MP3 files)."
    exit 1
}

if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $importer)) {
    Write-Log 'Stopped: Python environment or importer is missing.'
    exit 1
}

& $python $importer $source --output $output
if ($LASTEXITCODE -ne 0) {
    Write-Log "Stopped: importer failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Log "Completed: manifests created from $clipCount MP3 files; disabling scheduled task."
schtasks.exe /Change /TN $taskName /Disable | Out-Null
