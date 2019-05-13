#!/bin/sh

set -e 

# Find out whether ImageMagick(convert) is installed on the system.
CONVERT=convert
command -v $CONVERT >/dev/null 2>&1 || {
    echo >&2 "This script requires either convert installed."
    exit 1
}

if [ $# -ne 3 ]; then
    echo "Usage: ./imageProcessin <option> <directory_name_in_images> <output_folder_name>"
    exit 1
fi

IFS=$'\n'

option=$1
dirname=$2
output=$3

# Create output folder
mkdir -p $output

if [ $option == "-fhd" ]; then
    name="FHD"
    counter=0
    for file in `find $dirname -maxdepth 1 -type f`; do
	counter=`expr $counter + 1`
	convert $file -crop '2880x1620+0+90' output.png
	convert -resize 1920x1080 output.png $output/$name-$counter.png
	echo "SUCESS!!:$file"
    done
elif [ $option == "-hd" ]; then
    name="HD"
    counter=0
    for file in `find $dirname -maxdepth 1 -type f`; do
	counter=`expr $counter + 1`
	convert $file -crop '2880x1620+0+90' output.png
	convert -resize 1280x720 output.png $output/$name-$counter.png
	echo "SUCESS!!:$file"
    done
else 
    echo "Unknown option."
    exit 1
fi

rm -f output.png
