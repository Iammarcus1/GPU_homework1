#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cfloat>

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================


#include <cuda_runtime.h>

// --------------------------- CUDA kernels ---------------------------

// conv2d kernel (no padding, stride=1, ksize=5)
// input: (in_c, in_h, in_w)  contiguous: channel major
// weight: (out_c, in_c, k, k)
// bias: (out_c)
// output: (out_c, out_h, out_w)
__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias,
                              float* output,
                              int in_c, int out_c, int in_h, int in_w, int ksize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h - ksize + 1;
    int out_w = in_w - ksize + 1;
    int total = out_c * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    int oc = tmp / out_h;

    float sum = bias ? bias[oc] : 0.0f;

    // weight layout: oc * (in_c * ksize * ksize) + ic * (k*k) + kh * k + kw
    for (int ic = 0; ic < in_c; ++ic) {
        int in_chan_offset = ic * (in_h * in_w);
        int w_offset_base = oc * (in_c * ksize * ksize) + ic * (ksize * ksize);
        for (int kh = 0; kh < ksize; ++kh) {
            int ih = oh + kh;
            for (int kw = 0; kw < ksize; ++kw) {
                int iw = ow + kw;
                float a = input[in_chan_offset + ih * in_w + iw];
                float w = weight[w_offset_base + kh * ksize + kw];
                sum += a * w;
            }
        }
    }

    output[oc * (out_h * out_w) + oh * out_w + ow] = sum;
}

// maxpool2d kernel (ksize=2,stride=2), input: (c, in_h, in_w), output: (c, out_h, out_w)
__global__ void maxpool2d_kernel(const float* input, float* output,
                                 int c, int in_h, int in_w, int ksize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = in_h / ksize;
    int out_w = in_w / ksize;
    int total = c * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    int ch = tmp / out_h;

    float best = -1e30f;
    for (int kh = 0; kh < ksize; ++kh) {
        int ih = oh * ksize + kh;
        for (int kw = 0; kw < ksize; ++kw) {
            int iw = ow * ksize + kw;
            float v = input[ch * (in_h * in_w) + ih * in_w + iw];
            if (v > best) best = v;
        }
    }
    output[ch * (out_h * out_w) + oh * out_w + ow] = best;
}

// IFNode kernel (element-wise)
// input: current synaptic input (same shape as v_mem)
// v_mem: persistent membrane potentials (will be updated in place)
// spike_out: output spike (0 or 1) stored as float
// v_threshold: threshold value (e.g., 1.0f)
// reset_none: if true, do v -= v_threshold on spike; else set v = v_reset
__global__ void ifnode_kernel(const float* input, float* v_mem, float* spike_out,
                              int total, float v_threshold, float v_reset, bool reset_none)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = v_mem[idx] + input[idx];
    float spike = 0.0f;
    if (v >= v_threshold) {
        spike = 1.0f;
        if (reset_none) {
            v = v - v_threshold;
        } else {
            v = v_reset;
        }
    }
    v_mem[idx] = v;
    spike_out[idx] = spike;
}

// linear kernel (fully connected)
// input: (in_dim) or batched single sample contiguous
// weight: (out_dim, in_dim) flattened [out * in + in]
// bias: (out_dim)
// output: (out_dim)
__global__ void linear_kernel(const float* input, const float* weight, const float* bias,
                              float* output, int in_dim, int out_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim) return;

    const float* w_row = weight + idx * in_dim;
    float sum = bias ? bias[idx] : 0.0f;
    // dot product
    for (int j = 0; j < in_dim; ++j) {
        sum += w_row[j] * input[j];
    }
    output[idx] = sum;
}

// input: 输入特征图 (C_in x H_in x W_in)
// weights, bias: conv 参数
// v_mem: 膜电位
// output_pool: maxpool 输出
// ksize_pool: 池化大小
__global__ void conv_if_pool_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* v_mem,
    float* output_pool,
    int in_c, int out_c,
    int in_h, int in_w,
    int ksize_conv,
    int ksize_pool,
    float v_threshold,
    float v_reset,
    bool reset_none
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    int out_h = in_h - ksize_conv + 1;
    int out_w = in_w - ksize_conv + 1;
    int pool_h = out_h / ksize_pool;
    int pool_w = out_w / ksize_pool;

    if (ow >= pool_w || oh >= pool_h || oc >= out_c) return;

    float pool_val = -1e30f;

    // 对每个 pool 区域
    for (int ph = 0; ph < ksize_pool; ++ph) {
        for (int pw = 0; pw < ksize_pool; ++pw) {
            int conv_oh = oh * ksize_pool + ph;
            int conv_ow = ow * ksize_pool + pw;

            float conv_val = bias[oc];
            // 卷积
            for (int ic = 0; ic < in_c; ++ic) {
                for (int kh = 0; kh < ksize_conv; ++kh) {
                    for (int kw = 0; kw < ksize_conv; ++kw) {
                        int ih = conv_oh + kh;
                        int iw = conv_ow + kw;
                        float w = weights[oc*in_c*ksize_conv*ksize_conv + ic*ksize_conv*ksize_conv + kh*ksize_conv + kw];
                        conv_val += input[ic*in_h*in_w + ih*in_w + iw] * w;
                    }
                }
            }

            // IF 更新膜电位
            int idx_mem = oc*out_h*out_w + conv_oh*out_w + conv_ow;
            float v = v_mem[idx_mem] + conv_val;
            float spike = 0.0f;
            if (v >= v_threshold) {
                spike = 1.0f;
                if (reset_none) v -= v_threshold;
                else v = v_reset;
            }
            v_mem[idx_mem] = v;

            // 池化：取 IF spike 的 max
            if (spike > pool_val) pool_val = spike;
        }
    }

    // 写入池化输出
    output_pool[oc*pool_h*pool_w + oh*pool_w + ow] = pool_val;
}

// input_flat: flatten 后输入 (N_in)
// weights, bias: fc 参数
// v_mem: 膜电位
// spike_out: 输出 spike
__global__ void linear_if_kernel(
    const float* input_flat,
    const float* weights,
    const float* bias,
    float* output,
    float* v_mem,
    float* spike_out,
    int in_dim,
    int out_dim,
    float v_threshold,
    float v_reset,
    bool reset_none
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_dim) return;

    float sum = bias[oc];
    for (int ic = 0; ic < in_dim; ++ic) {
        sum += input_flat[ic] * weights[oc*in_dim + ic];
    }

    // IF 更新
    float v = v_mem[oc] + sum;
    float spike = 0.0f;
    if (v >= v_threshold) {
        spike = 1.0f;
        if (reset_none) v -= v_threshold;
        else v = v_reset;
    }

    v_mem[oc] = v;
    spike_out[oc] = spike;
    output[oc] = sum; // 如果需要 FC 输出累加 logits，可选
}

__global__ void add_vec_kernel(float* a, const float* b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] += b[idx];
}

// 批次卷积 + IF神经元 + 池化内核
__global__ void conv_if_pool_batch_kernel(
    const float* input, const float* weight, const float* bias,
    float* v_mem, float* output,
    int in_channels, int out_channels, int in_h, int in_w,
    int kernel_size, int pool_k,
    float v_threshold, float v_reset, bool reset_none,
    int batch_size)
{
    // 输出特征图尺寸（池化前）
    const int out_h = in_h - kernel_size + 1;
    const int out_w = in_w - kernel_size + 1;
    // 池化后尺寸
    const int pool_out_h = out_h / pool_k;
    const int pool_out_w = out_w / pool_k;
    
    // 线程对应的空间位置和通道
    const int pool_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pool_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % out_channels;  // 通道
    const int b = blockIdx.z / out_channels;  // 批次索引
    
    if (pool_x >= pool_out_w || pool_y >= pool_out_h || b >= batch_size) return;
    
    // 池化前的对应位置
    const int start_x = pool_x * pool_k;
    const int start_y = pool_y * pool_k;
    
    float max_val = -FLT_MAX;
    
    // 在池化区域内找最大值
    for (int py = 0; py < pool_k; ++py) {
        for (int px = 0; px < pool_k; ++px) {
            const int out_x = start_x + px;
            const int out_y = start_y + py;
            
            if (out_x < out_w && out_y < out_h) {
                // 计算卷积
                float sum = 0.0f;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            const int in_x = out_x + kx;
                            const int in_y = out_y + ky;
                            
                            if (in_x < in_w && in_y < in_h) {
                                // 批次索引计算
                                const int input_idx = b * in_channels * in_h * in_w + 
                                                    ic * in_h * in_w + 
                                                    in_y * in_w + in_x;
                                const int weight_idx = c * in_channels * kernel_size * kernel_size + 
                                                     ic * kernel_size * kernel_size + 
                                                     ky * kernel_size + kx;
                                
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                // 添加偏置
                sum += bias[c];
                
                // IF神经元模型
                const int v_idx = b * out_channels * out_h * out_w + 
                                c * out_h * out_w + 
                                out_y * out_w + out_x;
                
                float& v = v_mem[v_idx];
                v += sum;  // 累加膜电位
                
                float spike = 0.0f;
                if (v >= v_threshold) {
                    spike = 1.0f;
                    if (reset_none) {
                        v -= v_threshold;  // 软重置
                    } else {
                        v = v_reset;  // 硬重置
                    }
                }
                
                // 更新最大值
                if (spike > max_val) {
                    max_val = spike;
                }
            }
        }
    }
    
    // 写入池化输出
    if (max_val > -FLT_MAX) {
        const int output_idx = b * out_channels * pool_out_h * pool_out_w + 
                             c * pool_out_h * pool_out_w + 
                             pool_y * pool_out_w + pool_x;
        output[output_idx] = max_val;
    }
}

// 批次全连接 + IF神经元内核
__global__ void linear_if_batch_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* v_mem, float* spike_out,
    int in_features, int out_features,
    float v_threshold, float v_reset, bool reset_none,
    int batch_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_features * batch_size;
    
    if (idx >= total_elements) return;
    
    const int b = idx / out_features;  // 批次索引
    const int n = idx % out_features;  // 输出特征索引
    
    float sum = 0.0f;
    
    // 矩阵向量乘法
    for (int i = 0; i < in_features; ++i) {
        const int input_idx = b * in_features + i;
        const int weight_idx = n * in_features + i;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    // 添加偏置
    sum += bias[n];
    
    // IF神经元模型
    const int v_idx = b * out_features + n;
    float& v = v_mem[v_idx];
    v += sum;  // 累加膜电位
    
    float spike = 0.0f;
    if (v >= v_threshold) {
        spike = 1.0f;
        if (reset_none) {
            v -= v_threshold;  // 软重置
        } else {
            v = v_reset;  // 硬重置
        }
    }
    
    // 写入输出
    output[v_idx] = sum;  // 原始输出（如果需要）
    spike_out[v_idx] = spike;  // 脉冲输出
}

// 批次全连接内核（无IF神经元）
__global__ void linear_batch_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int in_features, int out_features,
    int batch_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_features * batch_size;
    
    if (idx >= total_elements) return;
    
    const int b = idx / out_features;  // 批次索引
    const int n = idx % out_features;  // 输出特征索引
    
    float sum = 0.0f;
    
    // 矩阵向量乘法
    for (int i = 0; i < in_features; ++i) {
        const int input_idx = b * in_features + i;
        const int weight_idx = n * in_features + i;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    // 添加偏置
    sum += bias[n];
    
    // 写入输出
    output[idx] = sum;
}

// 新增：flatten kernel（将 pool2 展平到 fc1 输入）
// ======================
__global__ void flatten_pool2_to_fc1_in_kernel(const float* pool2, float* fc1_in,
                                               int batch_size, int c, int h, int w) {
    // 每个线程处理一个元素 (b, idx)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * c * h * w;
    if (idx >= total) return;
    // 计算位置
    int tmp = idx;
    int b = tmp / (c * h * w); tmp = tmp % (c * h * w);
    // 这里保持与原来 flatten 顺序一致：channel-major within each sample
    fc1_in[idx] = pool2[idx];
}

// *** MODIFIED: 将 add_vec_batch_kernel 内联到 linear_batch_kernel ***
__global__ void linear_batch_kernel_with_acc(
    const float* input,
    const float* weight,
    const float* bias,
    float* output_acc, // 直接累加到 logits_acc
    int in_features, int out_features, int batch_size)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total)
    {
    int b = idx / out_features;
    int o = idx % out_features;
    float sum = 0.0f;
    const float* w_row = weight + o * in_features;
    const float* x_row = input + b * in_features;
    for (int i = 0; i < in_features; ++i)
    sum += w_row[i] * x_row[i];
    sum += bias[o];
    // *** 内联累加逻辑 ***
    output_acc[b * out_features + o] += sum;
    }
}
// --------------------------- End kernels ---------------------------

std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters (already allocated & copied in main)
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b,
    int batch_size = 1024)
{
    std::vector<int> predictions;
    const int num_images = static_cast<int>(images.size());
    predictions.resize(num_images);
    if (num_images == 0) return predictions;

    // Network dims (same as 你的原始实现)
    const int T = 4; // 时间步
    const float v_threshold = 1.0f;
    const bool reset_none = false;
    const float v_reset_unused = 0.0f;

   // Layer shapes (Conv1 out_c = 8, Conv2 out_c = 18, padding=0)
    const int c1_in_c = 1,  c1_out_c = 8, c1_k = 3, c1_in_h = 28, c1_in_w = 28;
    const int c1_out_h = c1_in_h - c1_k + 1; // 26
    const int c1_out_w = c1_in_w - c1_k + 1; // 26
    const int p1_c = c1_out_c, p1_h = c1_out_h / 2, p1_w = c1_out_w / 2; // 13, 13

    const int c2_in_c = c1_out_c, c2_out_c = 18, c2_k = 3, c2_in_h = p1_h, c2_in_w = p1_w;
    const int c2_out_h = c2_in_h - c2_k + 1; // 11
    const int c2_out_w = c2_in_w - c2_k + 1; // 11
    const int p2_c = c2_out_c, p2_h = c2_out_h / 2, p2_w = c2_out_w / 2; // 5, 5

    const int fc1_in = p2_c * p2_h * p2_w; // 18*5*5=450
    const int fc1_out = 128;
    const int fc2_out = 64;
    const int fc3_out = 10;

    // element counts (single sample)
    const int input_size = c1_in_c * c1_in_h * c1_in_w;
    const int conv1_size = c1_out_c * c1_out_h * c1_out_w;
    const int pool1_size = p1_c * p1_h * p1_w;
    const int conv2_size = c2_out_c * c2_out_h * c2_out_w;
    const int pool2_size = p2_c * p2_h * p2_w; // equals fc1_in
    const int fc1_size = fc1_out;
    const int fc2_size = fc2_out;
    const int fc3_size = fc3_out;

    // actual batch allocation (一次性按最大批次分配并复用)
    const int actual_batch_size = std::min(batch_size, num_images);

    // ================================
    // *** MODIFIED: 分配 pinned host buffer ***
    // ================================
    float* host_batch_input = nullptr;
    size_t host_batch_bytes = static_cast<size_t>(actual_batch_size) * input_size * sizeof(float);
    checkCudaErrors(cudaHostAlloc((void**)&host_batch_input, host_batch_bytes, cudaHostAllocDefault)); // *** MODIFIED

    // ================================
    // *** MODIFIED: 一次性在 device 分配并复用所有批次 buffer ***
    // ================================
    float *d_batch_input = nullptr, *d_batch_conv1 = nullptr, *d_batch_if1_v = nullptr, *d_batch_if1_spike = nullptr, *d_batch_pool1 = nullptr;
    float *d_batch_conv2 = nullptr, *d_batch_if2_v = nullptr, *d_batch_if2_spike = nullptr, *d_batch_pool2 = nullptr;
    float *d_batch_fc1_in = nullptr, *d_batch_fc1 = nullptr, *d_batch_if3_v = nullptr, *d_batch_if3_spike = nullptr;
    float *d_batch_fc2 = nullptr, *d_batch_if4_v = nullptr, *d_batch_if4_spike = nullptr;
    float *d_batch_fc3 = nullptr, *d_batch_logits_acc = nullptr;

    auto alloc_batch = [actual_batch_size](float** ptr, size_t single_elem_count) {
        size_t bytes = single_elem_count * static_cast<size_t>(actual_batch_size) * sizeof(float);
        checkCudaErrors(cudaMalloc(ptr, bytes));
    };

    alloc_batch(&d_batch_input, input_size);
    alloc_batch(&d_batch_conv1, conv1_size);
    alloc_batch(&d_batch_if1_v, conv1_size);
    alloc_batch(&d_batch_if1_spike, conv1_size);
    alloc_batch(&d_batch_pool1, pool1_size);
    alloc_batch(&d_batch_conv2, conv2_size);
    alloc_batch(&d_batch_if2_v, conv2_size);
    alloc_batch(&d_batch_if2_spike, conv2_size);
    alloc_batch(&d_batch_pool2, pool2_size);
    alloc_batch(&d_batch_fc1_in, pool2_size); // fc1_in == pool2 flattened
    alloc_batch(&d_batch_fc1, fc1_size);
    alloc_batch(&d_batch_if3_v, fc1_size);
    alloc_batch(&d_batch_if3_spike, fc1_size);
    alloc_batch(&d_batch_fc2, fc2_size);
    alloc_batch(&d_batch_if4_v, fc2_size);
    alloc_batch(&d_batch_if4_spike, fc2_size);
    alloc_batch(&d_batch_fc3, fc3_size);
    alloc_batch(&d_batch_logits_acc, fc3_size);

    const int BS = 256;

    // 主循环：按 actual_batch_size 分批处理
    for (int batch_start = 0; batch_start < num_images; batch_start += actual_batch_size) {
        const int current_batch_size = std::min(actual_batch_size, num_images - batch_start);

        // *** MODIFIED: 复用 pinned host buffer，只填充 current_batch_size 部分 ***
        for (int i = 0; i < current_batch_size; ++i) {
            int img_idx = batch_start + i;
            if (images[img_idx].size() == static_cast<size_t>(input_size)) {
                std::copy(images[img_idx].begin(), images[img_idx].end(), host_batch_input + i * input_size);
            } else {
                // 若尺寸不匹配则填 0（或按需处理）
                std::fill(host_batch_input + i * input_size, host_batch_input + (i + 1) * input_size, 0.0f);
            }
        }

        // H2D：拷贝当前批次数据（current_batch_size * input_size）
        checkCudaErrors(cudaMemcpy(d_batch_input, host_batch_input, static_cast<size_t>(current_batch_size) * input_size * sizeof(float), cudaMemcpyHostToDevice));

        // *** MODIFIED: 仅对当前批次需要清零的设备区域做 memset，避免清零整个分配区 ***
        checkCudaErrors(cudaMemset(d_batch_if1_v, 0, static_cast<size_t>(current_batch_size) * conv1_size * sizeof(float)));
        checkCudaErrors(cudaMemset(d_batch_if2_v, 0, static_cast<size_t>(current_batch_size) * conv2_size * sizeof(float)));
        checkCudaErrors(cudaMemset(d_batch_if3_v, 0, static_cast<size_t>(current_batch_size) * fc1_size * sizeof(float)));
        checkCudaErrors(cudaMemset(d_batch_if4_v, 0, static_cast<size_t>(current_batch_size) * fc2_size * sizeof(float)));
        checkCudaErrors(cudaMemset(d_batch_logits_acc, 0, static_cast<size_t>(current_batch_size) * fc3_size * sizeof(float)));

        // 时间步循环
        for (int t = 0; t < T; ++t) {
            // ---------- conv1 + IF + pool1 批次 kernel ----------
            dim3 block1(8, 8);
            dim3 grid1((p1_w + block1.x - 1) / block1.x,
                       (p1_h + block1.y - 1) / block1.y,
                       c1_out_c * current_batch_size);

            conv_if_pool_batch_kernel<<<grid1, block1>>>(
                d_batch_input, d_conv1_w, d_conv1_b, d_batch_if1_v, d_batch_pool1,
                c1_in_c, c1_out_c, c1_in_h, c1_in_w,
                c1_k, 2,
                v_threshold, v_reset_unused, reset_none,
                current_batch_size
            );
            checkCudaErrors(cudaGetLastError());

            // ---------- conv2 + IF + pool2 批次 kernel ----------
            dim3 block2(8, 8);
            dim3 grid2((p2_w + block2.x - 1) / block2.x,
                       (p2_h + block2.y - 1) / block2.y,
                       c2_out_c * current_batch_size);

            conv_if_pool_batch_kernel<<<grid2, block2>>>(
                d_batch_pool1, d_conv2_w, d_conv2_b, d_batch_if2_v, d_batch_pool2,
                c2_in_c, c2_out_c, c2_in_h, c2_in_w,
                c2_k, 2,
                v_threshold, v_reset_unused, reset_none,
                current_batch_size
            );
            checkCudaErrors(cudaGetLastError());

            // ---------- flatten（使用 kernel 替代 device->device memcpy） ----------
            int total_flat = current_batch_size * pool2_size; // pool2_size == fc1_in
            int threads = 256;
            int blocks = (total_flat + threads - 1) / threads;
            flatten_pool2_to_fc1_in_kernel<<<blocks, threads>>>(d_batch_pool2, d_batch_fc1_in, current_batch_size, p2_c, p2_h, p2_w);
            checkCudaErrors(cudaGetLastError());

            // ---------- fc1 (linear + IF) ----------
            linear_if_batch_kernel<<<(fc1_out * current_batch_size + BS - 1) / BS, BS>>>(
                d_batch_fc1_in, d_fc1_w, d_fc1_b, d_batch_fc1, d_batch_if3_v, d_batch_if3_spike,
                fc1_in, fc1_out, v_threshold, v_reset_unused, reset_none,
                current_batch_size
            );
            checkCudaErrors(cudaGetLastError());

            // ---------- fc2 (linear + IF) ----------
            linear_if_batch_kernel<<<(fc2_out * current_batch_size + BS - 1) / BS, BS>>>(
                d_batch_if3_spike, d_fc2_w, d_fc2_b, d_batch_fc2, d_batch_if4_v, d_batch_if4_spike,
                fc1_out, fc2_out, v_threshold, v_reset_unused, reset_none,
                current_batch_size
            );
            checkCudaErrors(cudaGetLastError());

            // ---------- fc3 (linear) 并内联累加到 logits_acc（减少一次 kernel 启动） ----------
            int total_fc3_elems = current_batch_size * fc3_out;
            dim3 block_fc3(256);
            dim3 grid_fc3((total_fc3_elems + block_fc3.x - 1) / block_fc3.x);
            linear_batch_kernel_with_acc<<<grid_fc3, block_fc3>>>(
                d_batch_if4_spike, d_fc3_w, d_fc3_b, d_batch_logits_acc,
                fc2_out, fc3_out, current_batch_size
            );
            checkCudaErrors(cudaGetLastError());
        } // end T

        // T 步平均 + argmax
        std::vector<float> host_batch_logits(static_cast<size_t>(current_batch_size) * fc3_out);
        checkCudaErrors(cudaMemcpy(host_batch_logits.data(), d_batch_logits_acc, static_cast<size_t>(current_batch_size) * fc3_out * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < current_batch_size; ++i) {
            int img_idx = batch_start + i;
            float* img_logits = host_batch_logits.data() + i * fc3_out;
            for (int k = 0; k < fc3_out; ++k) img_logits[k] /= T;
            int pred = 0;
            float best = img_logits[0];
            for (int k = 1; k < fc3_out; ++k) {
                if (img_logits[k] > best) { best = img_logits[k]; pred = k; }
            }
            predictions[img_idx] = pred;
        }
    } // end batches

    // 释放 device buffers（与分配顺序对应）
    cudaFree(d_batch_input); cudaFree(d_batch_conv1); cudaFree(d_batch_if1_v);
    cudaFree(d_batch_if1_spike); cudaFree(d_batch_pool1);
    cudaFree(d_batch_conv2); cudaFree(d_batch_if2_v); cudaFree(d_batch_if2_spike);
    cudaFree(d_batch_pool2);
    cudaFree(d_batch_fc1_in); cudaFree(d_batch_fc1); cudaFree(d_batch_if3_v);
    cudaFree(d_batch_if3_spike);
    cudaFree(d_batch_fc2); cudaFree(d_batch_if4_v); cudaFree(d_batch_if4_spike);
    cudaFree(d_batch_fc3); cudaFree(d_batch_logits_acc);

    // *** MODIFIED: 释放 pinned host memory ***
    if (host_batch_input) cudaFreeHost(host_batch_input); // *** MODIFIED

    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================