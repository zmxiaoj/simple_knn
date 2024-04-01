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

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  // 输入点的数目
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  // 初始化大小为P的全0张量
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  // 调用SimpleKNN::knn函数，计算每个点到其他点的距离
  // 输入点的数目、点的坐标tensor，输出的means张量
  // means保存每个点到最近3个点的距离平方的均值
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}