sudo apt update
sudo apt install -y build-essential cmake
git clone https://github.com/abria/TeraStitcher
mkdir build-terastitcher
cd build-terastitcher
cmake ../TeraStitcher/src
make -j `nproc`
sudo make install