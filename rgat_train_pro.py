import argparse
import json
import os
from datetime import datetime
import time
import dgl
import tqdm
from dgl.dataloading import GraphDataLoader
# from model_pro import RGCN_Model
from model_pro import RGAT_Model
from datasets_pro import eesg_datasets
import torch
import numpy as np
import random
from predict_pro import eval

def set_random_seed(seed_value=1234):
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def collate_fn_mini(batch):
    batch = list(zip(*batch))
    a_graphs = dgl.batch(batch[0])
    b_graphs = dgl.batch(batch[1])
    c_graphs = [dgl.batch(c_graph_batch) for c_graph_batch in list(zip(*batch[2]))]
    funcNames = batch[3]
    del batch
    return a_graphs, b_graphs, c_graphs, funcNames


def contra_loss_show(net, dataLoader, DEVICE, bs, f_strings="", f_gv="", f_ef=""):
    loss_val = []
    tot_cos = []
    tot_truth = []
    tq = tqdm.tqdm(dataLoader, ncols=80)
    for batch_id, batch in enumerate(tq, 1):

        cos_p, cos_n, loss = net.forward(batch, bs, f_strings, f_gv, f_ef)
        cos_p = list(cos_p.cpu().detach().numpy())
        cos_n = list(cos_n.cpu().detach().numpy())
        tot_cos += cos_p
        tot_truth += [1] * len(cos_p)
        tot_cos += cos_n
        tot_truth += [-1] * len(cos_n)
        loss_val.append(loss.item())
        tq.set_description("Eval:[" + str(loss.item()) + "]")
    return loss_val, 0.0, 0.0

def run(args,device):
    # set the save path
    set_random_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.cuda.set_device(args.gpu)
    base_path = "/mnt/data/leizhen/BinEnhance-main-1/"
    data_base = base_path + "EESG/"
    embedding_base = base_path + "baseline_embeddings/dataset2/" + str(args.fis) + "/"
    model_name = 'r-gat-' + str(args.modelname) + '-' + str(args.name) + '-' + str(args.max_edge_num) + '-' + str(
        args.max_node_num) + '-' + str(args.negative_rand) + '-' + str(args.lr) + '-' + str(
        args.num_layers) + '-' + str(args.sample_max) + "-" + str(args.batch_size)
    save_base = 'infonce/' + str(args.fis) + "/" + model_name
    train_func_embeddings_path = os.path.join(embedding_base,
                                              "train_function_embeddings_" + str(args.funcDim) + ".json")

    with open(train_func_embeddings_path, "r") as f:
        train_func_embeddings = json.load(f)
    test_func_embeddings_path = os.path.join(embedding_base, "test_function_embeddings_" + str(args.funcDim) + ".json")
    with open(test_func_embeddings_path, "r") as f:
        test_func_embeddings = json.load(f)
    valid_func_embeddings_path = os.path.join(embedding_base,
                                              "valid_function_embeddings_" + str(args.funcDim) + ".json")
    with open(valid_func_embeddings_path, "r") as f:
        valid_func_embeddings = json.load(f)

    train_strs_embeddings_path = os.path.join(os.path.join(base_path, "String_embedding"),
                                              "train_valid_strs_embeddings_" + str(args.funcDim) + ".json")
    with open(train_strs_embeddings_path, "r") as f:
        train_strs_embeddings = json.load(f)
    test_strs_embeddings_path = os.path.join(os.path.join(base_path, "String_embedding"),
                                             "test_strs_embeddings_" + str(args.funcDim) + ".json")
    with open(test_strs_embeddings_path, "r") as f:
        test_strs_embeddings = json.load(f)

    if not os.path.exists(save_base):
        os.makedirs(save_base)

    train_data_path = os.path.join(data_base, "train")
    valid_data_path = os.path.join(data_base, "valid")
    test_data_path = os.path.join(data_base, "test")


    f_strings_path = os.path.join(data_base, "all_strings.json")
    with open(f_strings_path, "r") as f:
        f_strings = json.load(f)

    f_ef_path = os.path.join(data_base, "all_external_funcs.json")
    with open(f_ef_path, "r") as f:
        f_ef = json.load(f)

    f_gv_path = os.path.join(data_base, "all_global_vars.json")
    with open(f_gv_path, "r") as f:
        f_gv = json.load(f)

    bs = args.batch_size
    train_dataset = eesg_datasets(train_data_path, funcs_embeddings=train_func_embeddings,
                                     strs_embeddings=train_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                     mode="train", args=args)
    valid_dataset = eesg_datasets(valid_data_path, funcs_embeddings=valid_func_embeddings,
                                     strs_embeddings=train_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                     mode="valid", args=args)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn_mini)
    test_dataset = eesg_datasets(test_data_path, funcs_embeddings=test_func_embeddings,
                                    strs_embeddings=test_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                    mode="test", args=args)


    print("""----Data statistics------'
              #Train samples %d
              #Valid samples %d
              #Test samples %d""" %
          (train_dataset.get_func_nums(), valid_dataset.get_func_nums(),
           test_dataset.get_func_nums()))
    print("mode_save:" + str(save_base))
    print("rel_type:" + str(train_dataset.rels))

    rels = train_dataset.rels
    funcDim = args.funcDim
    # model = RGCN_Model(funcDim, funcDim, funcDim, len(rels))
    model = RGAT_Model(funcDim, funcDim, funcDim, len(rels))
    model.to(device)
    print(model)

    # use optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)

    SAVE_FREQ = 5
    best_loss = 99999
    epoch = args.epoch
    patience = 5
    for i in range(epoch):
        loss_val = []
        tot_cos = []
        tot_truth = []
        time_start = time.time()
        model.train()
        p_n_gap = []
        train_dataset.shuffle()
        train_dataloader = GraphDataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn_mini)
        tq = tqdm.tqdm(train_dataloader, position=0, ncols=80)
        for batch in tq:
            cos_p, cos_n, loss = model.forward(batch, bs, f_strings, f_gv, f_ef)  

            cos_p = cos_p.cpu().detach().numpy()
            cos_n = cos_n.cpu().detach().numpy()
            p_n_gap.append(np.mean(cos_p - cos_n))
            cos_p = list(cos_p)
            cos_n = list(cos_n)
            tot_cos += cos_p
            tot_truth += [1] * len(cos_p)
            tot_cos += cos_n
            tot_truth += [-1] * len(cos_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy().item()
            loss_val.append(loss)
            tq.set_description("Train:EPOCH" + str(i) + "[" + str(loss) + "]")  # -cosp:[" + str(cos_p) + "]")

        print('Epoch: [%d]\tloss:%.4f\tp_n_gap:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
              (i, np.mean(loss_val), np.mean(p_n_gap), 0.0, datetime.now(), time.time() - time_start))
        model.eval()
        with torch.no_grad():
            time_start = time.time()

            loss_val, model_auc, tpr = contra_loss_show(model, valid_dataloader, args.gpu, bs, f_strings, f_gv, f_ef)

            print('Valid: [%d]\tloss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
                  (i, np.mean(loss_val), model_auc, datetime.now(), time.time() - time_start))

            patience -= 1
            if np.mean(loss_val) < best_loss:
                torch.save(model, save_base + "/RGAT-best.pt")
                best_loss = np.mean(loss_val)
                patience = 5

        if i % SAVE_FREQ == 0:
            torch.save(model, save_base + '/RGAT-' + str(i + 1) + ".pt")
        if patience <= 0:
            break

    save_res = base_path + "pro_results/" + args.fis + "/"
    eval_p = base_path + "pro_Eval_datas/"
    if not os.path.exists(save_res):
        os.makedirs(save_res)

    save_res += model_name.split("/")[-1] + "_test_func_embeddings.json"
    ans = eval(save_base, f_strings, f_gv, f_ef, hsg_data_path=test_data_path,
                 func_embedding_path=test_func_embeddings_path, relss_type=test_dataset.rels, args=args,
                 eval_type=0, str_embedding_path=test_strs_embeddings_path, model=args.fis, eval_p=eval_p, savepaths=save_res, device=device)
    return ans


if __name__ == '__main__':

    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--negative-slope', type=float, default=0.2)
    parser.add_argument('--name', type=str, default='dataset2')
    parser.add_argument('--modelname', type=str, default='InfoNCE')
    parser.add_argument("--funcDim", type=int, default=64)
    parser.add_argument("--max-edge-num", type=int, default=500)
    parser.add_argument("--max-node-num", type=int, default=999999)
    parser.add_argument("--sample-max", type=int, default=100000)
    parser.add_argument("--negative-rand", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--fis", type=str, default="Gemini")
    parser.add_argument("--have_str", type=bool, default=True, help="Whether to include string nodes in the subgraph")
    args = parser.parse_args()
    print(args)
    run(args, device)
    pass