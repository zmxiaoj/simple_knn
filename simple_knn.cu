/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

// 将32位整数x转换为morton码
__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	// 将x每个字节的最低2位复制到最高2位
	x = (x | (x << 16)) & 0x030000FF;
	// 将x每个字节的最低4位复制到最高4位
	x = (x | (x << 8)) & 0x0300F00F;
	// 将x每个字节的最低8位复制到最高8位
	x = (x | (x << 4)) & 0x030C30C3;
	// 将x每个字节的最低16位复制到最高16位
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

// 既可以在host上调用，也可以在device上调用
__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	// 将坐标映射到[0, 2^10 - 1]区间
	// 计算每个维度的morton码
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	// 每个thread对应一个点
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// 计算每个点的morton码
	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	// thread的idx
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	// 如果idx小于P，则me的minn和maxx初始化为当前点的坐标
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	// 如果idx大于P，则me的minn和maxx初始化为FLT_MAX和-FLT_MAX
	// FLT_MAX表示float的最大值，-FLT_MAX表示float的最小值
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	// 创建BOX_SIZE大小的共享内存，用于存储block内每个thread的MinMax
	__shared__ MinMax redResult[BOX_SIZE];

	// 并行归并
	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		// 当前thread的idx 小于 off，则将当前thread的MinMax保存到共享内存中
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		// block内thread同步
		__syncthreads();
		// 当前thread的idx 小于 off，则将当前thread的MinMax和off后的thread的MinMax进行归并
		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		// block内thread同步
		__syncthreads();
	}
	// 最后idx为0的thread中保存了block内归并的结果
	if (threadIdx.x == 0)
		// 将block对应的结果保存到boxes中
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	// 计算当前点point和参考点ref的距离
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	// 计算距离的平方
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	// 如果当前点的距离小于knn中的最大值，则更新knn
	// 将当前点的距离插入到knn中
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	// 每个thread对应一个点
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	
	// 取出当前点的坐标
	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	// 遍历当前点的前后各3个点
	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		// 更新关于当前点point最近的3个点的距离平方
		updateKBest<3>(point, points[indices[i]], best);
	}
	// 记录当前点的最近3个点距离平方的最大值
	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	// 遍历boxes中的每个box，找到最近的box，再找到最近的3个点
	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		// 计算box和当前点的距离
		float dist = distBoxPoint(box, point);
		// 如果box和当前点的距离大于reject或者大于best[2]，则跳过
		if (dist > reject || dist > best[2])
			continue;
		// 遍历box中的每个点
		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			// 更新关于当前点point最近的3个点的距离平方
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	// 计算当前点的3个最近点距离平方的均值
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

/**
 * @brief host函数
 * 
 * @param P [in] int, 输入点的数量
 * @param points [in] tensor, 输入点的坐标
 * @param meanDists [out] tensor, 输出点的均值距离
 */
void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	// 在GPU上分配内存，存储float3数据
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	// 计算cub的归约函数需要的临时内存大小
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	// 分配内存
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	// 实际调用cub的归约函数
	// 通过cub的归约函数计算points中每个维度的最小值，保存在result中
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	// 将最小值从GPU拷贝到CPU
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	// 通过cub的归约函数计算points中每个维度的最大值，保存在result中
	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	// 将最大值从GPU拷贝到CPU
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	// 设备向量device_vector，在GPU内存中分配，可以直接在cuda核函数中使用
	// 分配P个uint32_t的内存，用于存储morton码
	thrust::device_vector<uint32_t> morton(P);
	// 分配P个uint32_t的内存，用于存储排序后的morton码
	thrust::device_vector<uint32_t> morton_sorted(P);
	// 调用kernel函数，计算每个点的morton码
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	// 创建P大小的uint32_t向量，用于存储点的idx
	thrust::device_vector<uint32_t> indices(P);
	// 初始化indices为[0, 1, 2, ..., P-1]
	thrust::sequence(indices.begin(), indices.end());
	// 创建P大小的uint32_t向量，用于存储排序后的点的idx
	thrust::device_vector<uint32_t> indices_sorted(P);

	// 计算cub的排序函数需要的临时内存大小
	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	// 实际调用cub的基数排序函数
	// key - morton码，value - indices
	// 将morton码和indices按照morton码排序
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// 初始化boxes数目
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	// 创建num_boxes大小的MinMax向量，用于存储每个box的minn和maxx
	thrust::device_vector<MinMax> boxes(num_boxes);
	// 调用kernel函数，计算每个box的minn和maxx
	// grid大小为num_boxes，block大小为BOX_SIZE(1024)
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	// 调用kernel函数，计算每个点最近的3个点的距离平方的均值
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	// 释放GPU内存
	cudaFree(result);
}