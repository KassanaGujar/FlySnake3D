@echo off
REM Double-click to continue the stage 6 runs (separation test + 2 repeats).
REM Picks up where it stopped. Close this window to pause again; progress up to the
REM last finished generation is kept. Progress log: results\stage6_queue.log
cd /d "%~dp0"
set PY=C:/Users/User/.venvs/flysnake3d/Scripts/python
echo Stage 6 runs continuing. Close this window to pause.
"C:\Program Files\Git\bin\bash.exe" scripts/stage6_queue.sh
echo Done. See results\stage6_queue.log
pause
