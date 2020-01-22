terastitcher --test --projin="xml_import.xml" --imout_depth=16 --sparse_data
echo "test_middle_slice.tif saved to current working directory. Check that this looks right."
read -n 1 -p "Should we continue stitching? ([y]/n): " continue_stitching 
if "$continue_stitching" == "y"; then
	mpirun -n `nproc` python3 ~/Parastitcher_for_py37.py -2 --projin="xml_import.xml" --projout="xml_displcomp.xml" --sV=30 --sH=30 --sD=10 --subvoldim=60 --sparse_data --exectimes --exectimesfile="t_displcomp"
	terastitcher --displproj --projin="xml_displcomp.xml" --projout="xml_displproj.xml" --sparse_data
	terastitcher --displthres --projin="xml_displproj.xml" --projout="xml_displthres.xml" --threshold=0.3 --sparse_data
	terastitcher --placetiles --projin="xml_displthres.xml"
	mpirun -n 95 python3 ~/paraconverter2_3_2_py37.py -s="xml_merging.xml" -d="/home/ubuntu/stitched_data_stripcorrected" --sfmt="TIFF (unstitched, 3D)" --dfmt="TIFF (series, 2D)" --height=10480 --width=7184 --depth=171
else
	echo "Stitching will now end."
fi

