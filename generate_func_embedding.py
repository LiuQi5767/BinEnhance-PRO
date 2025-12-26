import argparse
import os
import json
import torch
import dgl
import numpy as np
from tqdm import tqdm
from predict_pro import eval
from model_pro import RGAT_Model
from datasets_pro import eesg_datasets
def load_embeddings(path):
    """Helper function to load embeddings from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def evaluate_with_poolsize(args, pool_sizes, device):
    """Run evaluation for multiple pool sizes and record MAP values."""
    # Paths
    base_path = "/home/leizhen/BinEnhance-main-1/"
    data_base = base_path + "EESG/"
    embedding_base = base_path + "baseline_embeddings/dataset2/" + str(args.fis) + "/"
    model_name = 'r-gat-' + str(args.modelname) + '-' + str(args.name) + '-' + str(args.max_edge_num) + '-' + str(
        args.max_node_num) + '-' + str(args.negative_rand) + '-' + str(args.lr) + '-' + str(
        args.num_layers) + '-' + str(args.sample_max) + "-" + str(args.batch_size)
    save_base = 'infonce/' + str(args.fis) + "/" + model_name

    # File paths
    test_func_embeddings_path = os.path.join(embedding_base, "test_function_embeddings_" + str(args.funcDim) + ".json")
    test_strs_embeddings_path = os.path.join(base_path, "String_embedding", "test_strs_embeddings_" + str(args.funcDim) + ".json")
    f_strings_path = os.path.join(data_base, "all_strings.json")
    f_gv_path = os.path.join(data_base, "all_global_vars.json")
    f_ef_path = os.path.join(data_base, "all_external_funcs.json")
    test_data_path = os.path.join(data_base, "test")

    # Load data
    test_func_embeddings = load_embeddings(test_func_embeddings_path)
    test_strs_embeddings = load_embeddings(test_strs_embeddings_path)
    f_strings = load_embeddings(f_strings_path)
    f_gv = load_embeddings(f_gv_path)
    f_ef = load_embeddings(f_ef_path)

    # Load dataset to get rel_type definitions
    test_dataset = eesg_datasets(test_data_path, funcs_embeddings=test_func_embeddings,
                                 strs_embeddings=test_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                 mode="test", args=args)

    # Evaluation output paths
    eval_p = base_path + "pro_Eval_datas/"
    save_res = base_path + "pro_results/" + args.fis + "/"
    if not os.path.exists(save_res):
        os.makedirs(save_res)

    save_res += model_name.split("/")[-1] + "_test_func_embeddings.json"

    # Evaluate for different pool sizes
    map_results = {}
    for poolsize in pool_sizes:
        print(f"Starting evaluation with poolsize={poolsize}...")
        ans = eval(save_base, f_strings, f_gv, f_ef, hsg_data_path=test_data_path,
                   func_embedding_path=test_func_embeddings_path, relss_type=test_dataset.rels, args=args,
                   eval_type=0, str_embedding_path=test_strs_embeddings_path, model=args.fis, poolsize=poolsize,
                   eval_p=eval_p, savepaths=save_res, device=device)
        map_results[poolsize] = ans
        print(f"Evaluation completed for poolsize={poolsize}. MAP: {ans}")

    return map_results

if __name__ == "__main__":

    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--negative-slope', type=float, default=0.2)
    parser.add_argument('--name', type=str, default='dataset2')
    parser.add_argument('--modelname', type=str, default='InfoNCE')
    parser.add_argument("--funcDim", type=int, default=128)
    parser.add_argument("--max-edge-num", type=int, default=500)
    parser.add_argument("--max-node-num", type=int, default=999999)
    parser.add_argument("--sample-max", type=int, default=100000)
    parser.add_argument("--negative-rand", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fis", type=str, default="Gemini")
    parser.add_argument("--have_str", type=bool, default=True, help="Whether to include string nodes in the subgraph")
    args = parser.parse_args()

    # Pool sizes to evaluate
    pool_sizes = [2, 16, 32, 128, 512, 1024, 2048, 4096, 8192, 10000]
    # Run evaluation
    map_results = evaluate_with_poolsize(args, pool_sizes, device)

    # Display results
    print("Final MAP results for different pool sizes:")
    for poolsize, map_value in map_results.items():
        print(f"Poolsize: {poolsize}, MAP: {map_value}")
