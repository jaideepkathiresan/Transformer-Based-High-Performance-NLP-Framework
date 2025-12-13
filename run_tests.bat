@echo off
echo Running HyperText Test Suite...
echo.
echo [1/2] Testing Core Ops...
python tests/test_ops.py
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo [2/2] Testing Models...
python tests/test_models.py
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo All Tests Passed!
