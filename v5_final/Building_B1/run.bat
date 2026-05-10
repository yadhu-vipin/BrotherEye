@echo off
echo Starting Building B1 DSTS Node...
echo.
start "Building B1 Server" cmd /k ""%~dp0..\..\.venv\Scripts\python.exe" building_server.py"
ping 127.0.0.1 -n 4 > nul
start "Building B1 Cameras + Track" cmd /c ""%~dp0..\..\.venv\Scripts\python.exe" simulate_cameras.py && echo. && echo Generating track plots... && "%~dp0..\..\.venv\Scripts\python.exe" "%~dp0..\plot_tracks.py" --building B1 && echo Done. Track plots saved. && pause"
echo.
echo Server started. Camera simulation will auto-generate track plots when done.
echo To query: "%~dp0..\..\.venv\Scripts\python.exe" query_network.py --person B1_Person_1
pause
