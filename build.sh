#!/bin/bash
# build.sh - 手动编译 TensorFlow Custom Op + DNN-Opt
#
# 使用方法:
#   ./build.sh                    # 编译所有
#   ./build.sh --dnnopt-only      # 仅编译 DNN-Opt
#   ./build.sh --ops-only         # 仅编译 Custom Op (假设 DNN-Opt 已编译)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DNNOPT_DIR="${DNNOPT_DIR:-/tmp/dnn-opt}"
DNNOPT_BUILD_DIR="${DNNOPT_BUILD_DIR:-${DNNOPT_DIR}/build}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "需要 Python 3"
        exit 1
    fi

    # 检查 TensorFlow
    if ! python3 -c "import tensorflow" 2>/dev/null; then
        log_error "需要安装 TensorFlow: pip install tensorflow"
        exit 1
    fi

    # 检查编译器
    if ! command -v g++ &> /dev/null; then
        log_error "需要 g++ 编译器"
        exit 1
    fi

    # 检查 DNN-Opt 源码
    if [ ! -d "${DNNOPT_DIR}" ]; then
        log_error "DNN-Opt 源码不存在: ${DNNOPT_DIR}"
        log_info "请先克隆 DNN-Opt: git clone https://github.com/xuefenghao5121/DNN-Opt.git ${DNNOPT_DIR}"
        exit 1
    fi

    log_info "依赖检查通过"
}

# 编译 DNN-Opt
build_dnnopt() {
    log_info "编译 DNN-Opt..."

    mkdir -p "${DNNOPT_BUILD_DIR}"
    cd "${DNNOPT_BUILD_DIR}"

    # 配置 CMake
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DDNNOPT_BUILD_TESTS=OFF \
        -DDNNOPT_BUILD_BENCHMARKS=OFF \
        -DDNNOPT_NATIVE_ARCH=ON \
        -DDNNOPT_USE_OPENMP=ON

    # 编译
    make -j$(nproc)

    # 检查输出
    if [ ! -f "${DNNOPT_BUILD_DIR}/src/libdnnopt_core.a" ]; then
        log_error "DNN-Opt 编译失败: libdnnopt_core.a 未生成"
        exit 1
    fi

    log_info "DNN-Opt 编译完成: ${DNNOPT_BUILD_DIR}/src/libdnnopt_core.a"
}

# 编译 TensorFlow Custom Op
build_custom_ops() {
    log_info "编译 TensorFlow Custom Op..."

    cd "${SCRIPT_DIR}"

    # 获取 TensorFlow 编译/链接标志
    TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
    TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

    log_info "TensorFlow compile flags: ${TF_CFLAGS}"
    log_info "TensorFlow link flags: ${TF_LFLAGS}"

    # 编译选项
    CXXFLAGS="-std=c++17 -O2 -fPIC -march=native -fopenmp"
    INCLUDES="-I${DNNOPT_DIR}/include"

    # 源文件
    SRCS="src/dnnopt_conv2d_op.cc src/dnnopt_matmul_op.cc"

    # 链接选项
    LDFLAGS="-shared"
    LIBS="-L${DNNOPT_BUILD_DIR}/src -ldnnopt_core -fopenmp -lpthread"

    # 编译
    log_info "编译命令:"
    log_info "g++ ${CXXFLAGS} ${TF_CFLAGS} ${INCLUDES} ${LDFLAGS} -o libdnnopt_ops.so ${SRCS} ${TF_LFLAGS} ${LIBS}"

    g++ ${CXXFLAGS} ${TF_CFLAGS} ${INCLUDES} ${LDFLAGS} \
        -o libdnnopt_ops.so \
        ${SRCS} \
        ${TF_LFLAGS} \
        ${LIBS} \
        -Wl,--whole-archive \
        -L${DNNOPT_BUILD_DIR}/src -ldnnopt_core \
        -Wl,--no-whole-archive

    if [ ! -f "libdnnopt_ops.so" ]; then
        log_error "Custom Op 编译失败"
        exit 1
    fi

    log_info "Custom Op 编译完成: ${SCRIPT_DIR}/libdnnopt_ops.so"
}

# 验证加载
verify_load() {
    log_info "验证 TensorFlow 加载..."

    python3 -c "
import tensorflow as tf
ops = tf.load_op_library('${SCRIPT_DIR}/libdnnopt_ops.so')
print('成功加载 Custom Op 库')
print('可用操作:')
for name in dir(ops):
    if not name.startswith('_'):
        print(f'  - {name}')
"

    if [ $? -eq 0 ]; then
        log_info "验证成功!"
    else
        log_error "验证失败"
        exit 1
    fi
}

# 主函数
main() {
    local build_dnnopt_flag=true
    local build_ops_flag=true
    local verify_flag=true

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dnnopt-only)
                build_ops_flag=false
                verify_flag=false
                shift
                ;;
            --ops-only)
                build_dnnopt_flag=false
                shift
                ;;
            --no-verify)
                verify_flag=false
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --dnnopt-only   仅编译 DNN-Opt"
                echo "  --ops-only      仅编译 Custom Op (假设 DNN-Opt 已编译)"
                echo "  --no-verify     跳过验证步骤"
                echo "  --help          显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done

    check_dependencies

    if [ "$build_dnnopt_flag" = true ]; then
        build_dnnopt
    fi

    if [ "$build_ops_flag" = true ]; then
        build_custom_ops
    fi

    if [ "$verify_flag" = true ]; then
        verify_load
    fi

    log_info "全部完成!"
}

main "$@"
