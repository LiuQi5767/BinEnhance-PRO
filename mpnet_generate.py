from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os
import json
from whitening_transformation import compute_kernel_bias, transform_and_normalize
import numpy as np

model = SentenceTransformer(r"/mnt/data/leizhen/BinEnhance-main/MPNet-base-v2")


def sentence_bert(argparse, model):
    dim = int(argparse.dimension)  # 确保 dim 是一个整数
    output_dir = argparse.output_dir
    save_path = os.path.join(output_dir, "strs_embeddings_" + str(dim) + ".json")
    data_path = argparse.input_dir
    kernel_path = os.path.join(output_dir,
                               "kernel_once_" + str(dim) + ".npy")  # this can use the same kernel and bias with WT
    bias_path = os.path.join(output_dir, "bias_once_" + str(dim) + ".npy")  # 白化变换所需的核矩阵和偏置向量的保存路径。
    strs_embeddings = {}

    model.to(0)  # 将模型移动到 GPU

    # 遍历输入目录中的 JSON 文件
    for file in os.listdir(data_path):
        filepath = os.path.join(data_path, file)
        with open(filepath, "r") as f:
            data = json.load(f)
        # 遍历 JSON 中的每个函数 fname 及其相关字符串 ts。
        for fname, ts in data.items():
            for t in ts:
                if t not in strs_embeddings:
                    # 删除前 9 个字符
                    strs_embeddings[t[9:]] = 0
                    # strs_embeddings[t] = 0

    # 转换嵌入
    str_texts = list(strs_embeddings.keys())
    str_embedding = model.encode(str_texts)

    # # 确保 str_embedding 是一个二维数组
    # if len(str_embedding.shape) == 1:
    #     str_embedding = np.expand_dims(str_embedding, axis=1)  # 如果是单维数组，转为二维数组

    # 白化变换
    if not os.path.exists(kernel_path):
        kernel, bias = compute_kernel_bias(str_embedding, dim)
    else:
        kernel = np.load(kernel_path)
        bias = np.load(bias_path)

    str_embedding = transform_and_normalize(str_embedding, kernel=kernel, bias=bias).tolist()

    # 保存核矩阵和偏置向量
    np.save(kernel_path, kernel)
    np.save(bias_path, bias)

    # 将嵌入保存到字典
    strs_embeddings = {}
    for i in range(len(str_texts)):
        strs_embeddings[str_texts[i]] = str_embedding[i]

    # 将嵌入保存到 json 文件
    with open(save_path, "w") as f:
        json.dump(strs_embeddings, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="the directory of input files")
    parser.add_argument("--dimension", "-d", type=str, required=True,
                        help="the output dimension of sentence embeddings")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="the directory of output files")

    args = parser.parse_args()
    sentence_bert(args, model)
