# Convert to .png
pdftoppm -png -r 300 instance_7_detection.pdf instance_7_detection

pdftoppm -png -r 300 instance_8_detection.pdf instance_8_detection

pdftoppm -png -r 300 instance_9_detection.pdf instance_9_detection

# Clip the image
convert instance_7_detection-1.png -shave 450x500 instance_7_detection.png

convert instance_8_detection-1.png -shave 450x500 instance_8_detection.png

convert instance_9_detection-1.png -shave 450x500 instance_9_detection.png

# Delete the intermediate files
rm instance_7_detection-1.png

rm instance_8_detection-1.png

rm instance_9_detection-1.png