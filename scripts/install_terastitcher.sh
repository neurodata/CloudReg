sudo apt update
sudo apt  install build-essential cmake
git clone https://github.com/abria/TeraStitcher
mkdir build-terastitcher
cd build-terastitcher
cmake ../TeraStitcher/src
make -j `nproc`
make install