@echo off
echo Installing Distributed DSTS Dependencies...
echo.
echo Step 1: Installing CMake (Required for dlib)...
pip install cmake
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install CMake. Please install Python and add to PATH.
    pause
    exit /b
)

echo.
echo Step 2: Installing core libraries (This may take a few minutes)...
pip install numpy scikit-learn face_recognition

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed! 
    echo Common fix: You need Visual Studio C++ Build Tools installed.
    echo Download it here: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    pause
) else (
    echo.
    echo [SUCCESS] All dependencies installed successfully!
    pause
)
