@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" labelingtool edit_mask ^
    --input-dir "%WORKSPACE%\data_dst\aligned" ^
    --confirmed-dir "%WORKSPACE%\data_dst\aligned_confirmed" ^
    --skipped-dir "%WORKSPACE%\data_dst\aligned_skipped"
