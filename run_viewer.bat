@echo off
rem FlySnake3D viewer launcher. Double-click, or run with options:
rem   run_viewer.bat                 game replay viewer (opens greedy A* replay)
rem   run_viewer.bat --brain         fly brain activity view
rem   run_viewer.bat --replay=results\replays\stage2\astar_greedy_seed1.jsonl
setlocal
cd /d "%~dp0"
if "%GODOT%"=="" set "GODOT=%USERPROFILE%\tools\godot-4.7.2\Godot_v4.7.2-stable_win64.exe"
if "%PY%"=="" set "PY=%USERPROFILE%\.venvs\flysnake3d\Scripts\python.exe"
if not exist "%GODOT%" (echo Godot not found at %GODOT% - set GODOT to its path & pause & exit /b 1)
if not exist "results\brain\looming_demo\activity.json" (
  echo Recording fly brain activity for the brain view, about 30 s, first run only...
  "%PY%" -m experiments.record_brain_activity
)
start "" "%GODOT%" --path viewer -- %*
