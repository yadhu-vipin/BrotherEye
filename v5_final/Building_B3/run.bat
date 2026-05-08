@echo off
echo Starting Building B3 DSTS Node...
echo.
start "Building B3 Server" cmd /k python building_server.py
timeout /t 3 >nul
start "Building B3 Cameras" cmd /k python simulate_cameras.py
echo.
echo Both processes started in separate windows.
echo To query: python query_network.py --person B1_Person_2
pause
