mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
      -DCPM_DOWNLOAD_ALL=ON -DCMAKE_DISABLE_FIND_PACKAGE_fmt=ON \
      -DCPM_SOURCE_CACHE=/workspace/.cpm_cache -GNinja ..
ninja

echo "Compilation Finish"
cd ..
rm -rf ../*.so
for file in $(find "./build" -maxdepth 1 -name "*.so"); do
    abs_file=$(realpath $file)
    if [ -e $abs_file ]; then
        ln -s $abs_file ../
        echo "Copied $abs_file..."
    fi
done