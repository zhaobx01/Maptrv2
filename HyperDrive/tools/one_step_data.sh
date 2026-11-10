#!/bin/bash
# 一键下载数据集、预训练模型、创建软链接、下载信息文件
# 调用方式：bash HyperDrive/tools/one_step_data.sh /data/datasets/nuscenes /data/datasets/MapTRv2 /workspace/MapTRv2

# 设置默认参数
NUSCENES_DATA_PATH=${1:-/data/datasets/nuscenes}
MAPTR_DATA_PATH=${2:-/data/datasets/MapTRv2}
WORKSPACE_PATH=${3:-/workspace/MapTRv2}
CACHE_PATH=${4:-null}  # 可选缓存路径，如/cfs/datasets/hyperdrive，下载时会优先在此目录下查找文件。

if [[ "${CACHE_PATH}" != "null" && ! -d "${CACHE_PATH}" ]]; then
    echo "⚠️ **注意：缓存路径不存在**: ${CACHE_PATH}"
    sleep 1
fi

echo "**=== 参数配置 ===**"
echo "**NuScenes数据路径**: $NUSCENES_DATA_PATH"
echo "**MapTR数据路径**: $MAPTR_DATA_PATH"
echo "**工作空间路径**: $WORKSPACE_PATH"
echo "**缓存路径（下载时会优先在此目录下查找文件）**: $CACHE_PATH"
echo ""

TEST_MODE=false
# 如果最后一个参数是 --test，则启用测试模式，否则启用完整模式
last_arg="${@: -1}"
if [ "$last_arg" = "--test" ]; then
    TEST_MODE=true
    echo "🧪 **测试模式已启用，将下载 mini 数据集**"
fi

# 检查必要的依赖
check_dependencies() {
    local missing=()
    if ! command -v aria2c &>/dev/null; then
        missing+=("aria2")
    fi
    if ! command -v zstd &>/dev/null; then
        missing+=("zstd")
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "❌ **错误: 缺少必要的依赖: ${missing[*]}**"
        echo "请先安装: apt-get install -y ${missing[*]}"
        exit 1
    fi
}
check_dependencies

# 根据地域选择 COS 域名
get_domain_name() {
    local zone region
    zone=$(curl -s --connect-timeout 2 http://169.254.10.10/latest/meta-data/placement/zone 2>/dev/null)
    if [[ -n "$zone" ]]; then
        # 按 "-" 分割，取前两部分组成地域字符串
        region=$(echo "$zone" | awk -F'-' '{print $1"-"$2}')
    fi

    case "$region" in
        ap-shanghai)
            echo "https://haihub-model-1251001002.cos.ap-shanghai.myqcloud.com"
            ;;
        ap-beijing)
            echo "https://haihub-model-bj-1251001002.cos.ap-beijing.myqcloud.com"
            ;;
        ap-guangzhou)
            echo "https://haihub-model-gz-1251001002.cos.ap-guangzhou.myqcloud.com"
            ;;
        ap-nanjing)
            echo "https://haihub-model-nj-1251001002.cos.ap-nanjing.myqcloud.com"
            ;;
        *)
            # default: shanghai
            echo "https://haihub-model-bj-1251001002.cos.ap-beijing.myqcloud.com"
            ;;
    esac
}

BASE_URL="$(get_domain_name)/HyperDrive"
echo "=== 当前使用的 BASE_URL: $BASE_URL ==="

check_directory() {
    [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]
}

safe_download_with_cache() {
    local url="$1" filename="$2"
    # 优先从缓存路径复制文件
    local return_code=0
    if [[ "$CACHE_PATH" != "null" && -d "$CACHE_PATH" ]]; then
        local cached_file="$CACHE_PATH/$(basename "$url")"
        cached_file="$(realpath -m "$cached_file")"  # 规范化路径，顺便避免双斜杠

        if [[ -f "$cached_file" ]]; then
            echo "**从缓存复制文件: $cached_file -> $filename **"
            rsync -a "$cached_file" "$filename"
            return_code=$?
        else
            return_code=1
            if safe_download "$url" "$filename"; then
                echo "**将下载的文件复制到缓存: $filename -> $cached_file **"
                rsync -a "$filename" "$cached_file"
                return_code=$?
            fi
        fi
    else
        safe_download "$url" "$filename"
        return_code=$?
    fi
    return $return_code
}

safe_download() {
    local url="$1" filename="$2" max_retries=3 retry_count=0

    while [[ $retry_count -lt $max_retries ]]; do
        echo "**下载尝试 $((retry_count + 1))/$max_retries**: $filename"
        rm -f "$filename"

        echo "**使用 aria2c 进行多线程下载**"
        if aria2c --enable-color=false --show-console-readout=false -x 8 -s 8 -k 1M -d "$(dirname "$filename")" -o "$(basename "$filename")" "$url" && [[ -f "$filename" ]]; then
            echo "✅ **下载完成**: $filename"
            return 0
        fi

        retry_count=$((retry_count + 1))
    done

    echo "❌ **下载失败**: $filename"
    return 1
}

# 下载和解压 NuScenes 数据集
download_nuscenes() {
    local target_path="$1"
    local nuscenes_url
    if [ "$TEST_MODE" = true ]; then
        nuscenes_url="$BASE_URL/nuscenes_dataset/nuscenes_hyperdrive_mini.tar.zst"
        echo "**开始准备 NuScenes mini 数据集**"
    else
        nuscenes_url="$BASE_URL/nuscenes_dataset/nuscenes_hyperdrive.tar.zst"
        echo "**开始准备 NuScenes 完整数据集**"
    fi

    local max_retries=3 retry_count=0
    mkdir -p "$target_path"

    while [ $retry_count -lt $max_retries ]; do
        echo "**下载尝试 $((retry_count + 1))/$max_retries -> 临时文件下载中**"
        tmpfile=$(mktemp /data/nuscenes.XXXXXX) || { echo "❌ 无法创建临时文件"; return 1; }

        # 使用 safe_download_with_cache 下载到临时文件
        if ! safe_download_with_cache "$nuscenes_url" "$tmpfile"; then
            echo "❌ **下载失败（safe_download_with_cache）**"
            rm -f "$tmpfile"
            retry_count=$((retry_count + 1))
            sleep 2
            continue
        fi

        # 剥离顶层目录
        echo "**开始解压到目标路径: $target_path**"
        tar_accelerate_args="--no-same-owner --no-same-permissions --touch"
        zstd -dc --no-progress "$tmpfile" | tar -xf - --strip-components=2 -C "$target_path" ${tar_accelerate_args}
        tar_exit=$?

        if [[ $tar_exit -eq 0 ]]; then
            echo "✅ **数据集处理完成**"
            rm -f "$tmpfile"
            return 0
        else
            echo "❌ **解压失败 (tar: $tar_exit)**，清理并重试"
            rm -f "$tmpfile"
            rm -rf "$target_path"/*
            retry_count=$((retry_count + 1))
            sleep 5
        fi
    done

    echo "❌ **数据集下载或解压失败，已达最大重试次数**"
    return 1
}

download_can_bus() {
    local target_dir="$1"
    local can_bus_url="$BASE_URL/nuscenes_dataset/can_bus.tar.gz"
    local max_retries=3 retry_count=0

    echo "**开始准备 can_bus 数据**"
    mkdir -p "$target_dir"

    while [ $retry_count -lt $max_retries ]; do
        echo "**下载尝试 $((retry_count + 1))/$max_retries -> 临时文件下载中**"
        tmpfile=$(mktemp /data/can_bus.XXXXXX) || { echo "❌ 无法创建临时文件"; return 1; }

        if ! safe_download_with_cache "$can_bus_url" "$tmpfile"; then
            echo "❌ **can_bus 下载失败（safe_download_with_cache）**"
            rm -f "$tmpfile"
            retry_count=$((retry_count + 1))
            sleep 2
            continue
        fi

    # 假定为 .tar.gz，使用 gzip 解压并剥离顶层目录
    tar -xzf "$tmpfile" --strip-components=1 -C "$target_dir"
        tar_exit=$?

        if [ $tar_exit -eq 0 ]; then
            echo "✅ **can_bus 数据处理完成**"
            rm -f "$tmpfile"
            return 0
        else
            echo "❌ **can_bus 解压失败 (tar: $tar_exit)**，清理并重试"
            rm -f "$tmpfile"
            rm -rf "$target_dir/can_bus" "$target_dir"/* 2>/dev/null || true
            retry_count=$((retry_count + 1))
            sleep 5
        fi
    done

    echo "❌ **can_bus 下载或解压失败，已达最大重试次数**"
    return 1
}

# setup_maptr_data 改为先下载 pkl 压缩包再解压（同样逻辑）
setup_maptr_data() {
    echo "**=== 设置MapTR数据 ===**"
    local target_dir="$1"

    mkdir -p "$target_dir"
    local pkl_url="$BASE_URL/nuscenes_dataset/maptr-pkl.tar.gz"

    tmpfile=$(mktemp /data/maptr_pkl.XXXXXX) || { echo "❌ 无法创建临时文件"; return 1; }

    if ! safe_download_with_cache "$pkl_url" "$tmpfile"; then
        echo "❌ **pkl 下载失败**"
        rm -f "$tmpfile"
        return 1
    fi

    # 假定为 .tar.gz，使用 gzip 解压
    tar -xzvf "$tmpfile" -C "$target_dir"
    tar_exit=$?
    rm -f "$tmpfile"

    if [ $tar_exit -ne 0 ]; then
        echo "❌ **pkl 解压失败 (tar: $tar_exit)**"
        return 1
    fi

    # 下载 ckpts 文件
    mkdir -p "$WORKSPACE_PATH/ckpts"
    safe_download "$BASE_URL/ckpt/resnet50-19c8e357.pth" "$WORKSPACE_PATH/ckpts/resnet50-19c8e357.pth" || { echo "❌ **resnet50 ckpt 下载失败**"; return 1; }

    echo "✅ **MapTR数据设置完成**"
}

# 创建软链接
create_symlink() {
    local source="$1" target="$2"

    if [ -L "$target" ]; then
        [ "$(readlink "$target")" = "$source" ] && return 0
        rm -f "$target"
    elif [ -e "$target" ]; then
        rm -rf "$target"
    fi

    ln -s "$source" "$target"
    echo "✅ **软链接创建**: $(basename "$target") -> $source"
}

# 创建NuScenes数据软链接
create_nuscenes_symlinks() {
    echo "**=== 创建NuScenes数据软链接 ===**"
    mkdir -p "$2" && cd "$2"

    # 处理可能的嵌套目录
    local source_path="$1"
    [ -d "$1/nuscenes" ] && source_path="$1/nuscenes"

    # 创建核心目录软链接
    for item in samples sweeps maps v1.0-trainval; do
        [ -e "$source_path/$item" ] && create_symlink "$source_path/$item" "$2/$item"
    done

    echo "✅ **NuScenes数据软链接创建完成**"
}

# **第一步：检查和下载NuScenes原始数据与can_bus数据**
echo "**=== 第一步：检查NuScenes原始数据 ===**"
if check_directory "$NUSCENES_DATA_PATH"; then
    echo "✅ **NuScenes数据集已存在**"
else
    start=$(date +%s)
    download_nuscenes "$NUSCENES_DATA_PATH" || { echo "❌ **NuScenes下载失败**"; exit 1; }
    end=$(date +%s)
    duration=$((end - start))
    echo "⏱️ **NuScenes数据集下载和解压完成，耗时 $duration 秒**"
fi

echo "**=== 检查can_bus数据 ===**"
parent_dir=$(dirname "$NUSCENES_DATA_PATH")
CAN_BUS_DATA_PATH="$parent_dir/can_bus"
if check_directory "$CAN_BUS_DATA_PATH"; then
    echo "✅ **can_bus数据集已存在**"
else
    download_can_bus "$CAN_BUS_DATA_PATH" || { echo "❌ **can_bus下载失败**"; exit 1; }
fi

# **第二步：设置MapTR数据**
echo ""
echo "**=== 第二步：设置MapTR数据 ===**"
setup_maptr_data "$MAPTR_DATA_PATH" || { echo "❌ **MapTR数据设置失败**"; exit 1; }

# **第三步：创建NuScenes数据软链接**
echo ""
create_nuscenes_symlinks "$NUSCENES_DATA_PATH" "$MAPTR_DATA_PATH" || { echo "❌ **软链接创建失败**"; exit 1; }

# **第四步：在工作空间创建软链接**
echo ""
echo "**=== 第四步：在工作空间创建软链接 ===**"
[ ! -d "$WORKSPACE_PATH" ] && { echo "❌ **工作空间不存在**: $WORKSPACE_PATH"; exit 1; }

mkdir -p "$WORKSPACE_PATH/data"
create_symlink "$MAPTR_DATA_PATH" "$WORKSPACE_PATH/data/nuscenes"
create_symlink "$CAN_BUS_DATA_PATH" "$WORKSPACE_PATH/data/can_bus"

echo ""
echo "🎉 **================================================================**"
echo "🎉 **所有步骤完成! MapTR环境设置成功!**"
echo "🎉 **================================================================**"
echo ""
echo "📊 **数据架构总览**:"
echo "├── 📁 **NuScenes原始数据**: $NUSCENES_DATA_PATH"
echo "├── 📁 **MapTR数据管理**: $MAPTR_DATA_PATH"
echo "│   ├── 🔗 samples -> $NUSCENES_DATA_PATH/samples"
echo "│   ├── 🔗 sweeps -> $NUSCENES_DATA_PATH/sweeps"
echo "│   ├── 🔗 maps -> $NUSCENES_DATA_PATH/maps"
echo "│   ├── 🔗 v1.0-trainval -> $NUSCENES_DATA_PATH/v1.0-trainval"
echo "│   └── 📄 *.pkl"
echo "└── 💼 **工作空间**: $WORKSPACE_PATH"
echo "    └── 🔗 data/nuscenes -> $MAPTR_DATA_PATH"
