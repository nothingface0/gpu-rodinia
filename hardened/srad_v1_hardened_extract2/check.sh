echo "**********************************", "checking std"
diff golden_stdout.txt ../srad_v1/golden_stdout.txt 
diff image_out.pgm ../srad_v1/golden_image_out.pgm
diff golden_stderr.txt ../srad_v1/golden_stderr.txt 
