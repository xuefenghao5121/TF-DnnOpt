#!/bin/bash
# build_stub.sh - 编译 TensorFlow Custom Op 使用 Stub 实现
#
# 用于 x86_64 平台测试，不需要真实的 DNN-Opt ARM 库
#
# 使用方法:
#   ./build_stub.sh              # 编译并验证
#   ./build_stub.sh --no-verify  # 仅编译，跳过验证

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

    # 检查 TensorFlow (从临时目录运行避免模块冲突)
    if ! (cd /tmp && python3 -c "import tensorflow") 2>&1 > /dev/null; then
        log_error "需要安装 TensorFlow: pip install tensorflow"
        exit 1
    fi

    # 检查编译器
    if ! command -v g++ &> /dev/null; then
        log_error "需要 g++ 编译器"
        exit 1
    fi

    log_info "依赖检查通过"
}

# 编译 TensorFlow Custom Op 使用 Stub
build_custom_ops_stub() {
    log_info "编译 TensorFlow Custom Op (使用 Stub 实现)..."

    cd "${SCRIPT_DIR}"

    # 获取 TensorFlow 编译/链接标志 (从临时目录运行避免模块冲突)
    TF_CFLAGS=$(cd /tmp && python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
    TF_LFLAGS=$(cd /tmp && python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

    log_info "TensorFlow compile flags: ${TF_CFLAGS}"
    log_info "TensorFlow link flags: ${TF_LFLAGS}"

    # 编译选项
    CXXFLAGS="-std=c++17 -O2 -fPIC -fopenmp"
    INCLUDES="-I${SCRIPT_DIR}/src"

    # 源文件 (包含 stub 实现)
    SRCS="src/dnnopt_stub_impl.cpp src/dnnopt_conv2d_op.cc src/dnnopt_matmul_op.cc"

    # 链接选项
    LDFLAGS="-shared"
    LIBS="-fopenmp -lpthread"

    # 编译
    log_info "编译命令:"
    log_info "g++ ${CXXFLAGS} ${TF_CFLAGS} ${INCLUDES} ${LDFLAGS} -o libdnnopt_ops.so ${SRCS} ${TF_LFLAGS} ${LIBS}"

    g++ ${CXXFLAGS} ${TF_CFLAGS} ${INCLUDES} ${LDFLAGS} \
        -o libdnnopt_ops.so \
        ${SRCS} \
        ${TF_LFLAGS} \
        ${LIBS}

    if [ ! -f "libdnnopt_ops.so" ]; then
        log_error "Custom Op 编译失败"
        exit 1
    fi

    log_info "Custom Op 编译完成: ${SCRIPT_DIR}/libdnnopt_ops.so"
}

# 验证加载
verify_load() {
    log_info "验证 TensorFlow 加载..."

    (cd /tmp && python3 -c "
import tensorflow as tf
ops = tf.load_op_library('${SCRIPT_DIR}/libdnnopt_ops.so')
print('成功加载 Custom Op 库')
print('可用操作:')
for name in dir(ops):
    if not name.startswith('_'):
        print(f'  - {name}')
")

    if [ $? -eq 0 ]; then
        log_info "验证成功!"
    else
        log_error "验证失败"
        exit 1
    fi
}

# 运行正确性测试
run_correctness_tests() {
    log_info "运行正确性测试..."

    (cd /tmp && python3 "${SCRIPT_DIR}/tests/test_matmul_correctness.py")
}

# 主函数
main() {
    local verify_flag=true
    local test_flag=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-verify)
                verify_flag=false
                shift
                ;;
            --test)
                test_flag=true
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --no-verify     跳过验证步骤"
                echo "  --test          运行正确性测试"
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
    build_custom_ops_stub

    if [ "$verify_flag" = true ]; then
        verify_load
    fi

    if [ "$test_flag" = true ]; then
        run_correctness_tests
    fi

    log_info "全部完成!"
}

main "$@"
