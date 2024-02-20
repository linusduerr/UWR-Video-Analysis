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

cd ..

python uwr_bg_test.py --algo GSOC

ffmpeg -i input/L.mp4 -map 0:a -acodec copy A.mp4
ffmpeg -i out.mp4 -i A.mp4 -c:v copy -map 0:v -map 1:a -y output/final.mp4

del A.mp4
del out.mp4