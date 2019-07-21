@echo off
echo Press space if you want to delete content in workspace folder
pause
call _internal\setenv.bat
mkdir "%WORKSPACE%" 2>nul
rmdir "%WORKSPACE%\data_src" /s /q 2>nul
mkdir "%WORKSPACE%\data_src" 2>nul
mkdir "%WORKSPACE%\data_src\aligned" 2>nul
rmdir "%WORKSPACE%\data_dst" /s /q 2>nul
mkdir "%WORKSPACE%\data_dst" 2>nul
mkdir "%WORKSPACE%\data_dst\aligned" 2>nul
rmdir "%WORKSPACE%\model" /s /q 2>nul
mkdir "%WORKSPACE%\model" 2>nul
echo DONE
pause