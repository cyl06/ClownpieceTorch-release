#!/bin/bash

set -e

OUTFILE="TEST3.md"
PYTEST="python3 test_performance.py"

echo "# 并行策略性能测试结果" > $OUTFILE

declare -A BUILD_CMDS
BUILD_CMDS["Sequential"]="g++ -O2 -Wall -shared -std=c++17 -fPIC \`python3 -m pybind11 --includes\` tensor.cc tensor_pybind.cc -o tensor_impl\`python3-config --extension-suffix\`"
BUILD_CMDS["ThreadPerOp"]="g++ -O2 -Wall -shared -std=c++17 -fPIC -DUSE_THREAD_PER_OP -pthread \`python3 -m pybind11 --includes\` tensor.cc tensor_pybind.cc -o tensor_impl\`python3-config --extension-suffix\`"
BUILD_CMDS["OpenMP"]="g++ -O2 -Wall -shared -std=c++17 -fPIC -DUSE_OPENMP -fopenmp \`python3 -m pybind11 --includes\` tensor.cc tensor_pybind.cc -o tensor_impl\`python3-config --extension-suffix\`"
BUILD_CMDS["ThreadPool"]="g++ -O2 -Wall -shared -std=c++17 -fPIC -DUSE_THREAD_POOL -pthread \`python3 -m pybind11 --includes\` tensor.cc tensor_pybind.cc -o tensor_impl\`python3-config --extension-suffix\`"

for STRATEGY in Sequential ThreadPerOp OpenMP ThreadPool; do
    echo -e "\n## $STRATEGY\n" >> $OUTFILE
    echo "编译 $STRATEGY ..."
    eval ${BUILD_CMDS[$STRATEGY]}
    echo "运行测试 $STRATEGY ..."
    $PYTEST >> $OUTFILE 2>&1
done

echo -e "\n所有测试完成，结果已写入 $OUTFILE"