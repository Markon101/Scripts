#!/bin/bash
for i in *.gif;
do
	ffmpeg -i "$i" -c:v h264 -b:v 4M "$(basename $i).mp4";
done
