@echo off
echo Starting Building B4 DSTS Node...
echo.
start "Building B4 Server" cmd /k python building_server.py
ping 127.0.0.1 -n 4 > nul
start "Building B4 Cameras" cmd /k python simulate_cameras.py
echo.
echo Both processes started in separate windows.
echo To query: python query_network.py --person B1_Person_2
pause
