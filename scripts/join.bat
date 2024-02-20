@echo off
cd ../input

(
    for %%i in (L*.mp4) do echo file '%%i'
) > listL.txt

(
    for %%i in (R*.mp4) do echo file '%%i'
) > listR.txt

ffmpeg -f concat -i listL.txt -c copy L.mp4
ffmpeg -f concat -i listR.txt -c copy R.mp4

del listL.txt
del listR.txt

echo Videos merged successfully.
