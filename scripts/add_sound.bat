@echo off

cd ..

ffmpeg -i input/L.mp4 -map 0:a -acodec copy A.mp4
ffmpeg -i out.mp4 -i A.mp4 -c:v copy -map 0:v -map 1:a -y output/final.mp4

del A.mp4
del out.mp4

echo Video with sound generated.