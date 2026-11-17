@echo off
REM Double-click to start or continue the stage 7 runs (smelling ANN, learning fly,
REM matched no-learning control, then the full evaluation).
REM Picks up where it stopped. Close this window to pause again; progress up to the
REM last finished generation is kept. Progress log: results\stage7_queue.log
cd /d "%~dp0"
set PY=C:/Users/User/.venvs/flysnake3d/Scripts/python
echo Stage 7 runs continuing. Close this window to pause.
"C:\Program Files\Git\bin\bash.exe" scripts/stage7_queue.sh
echo Done. See results\stage7_queue.log
pause
