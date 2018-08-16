import os
import pickle
import torch
import data
import numpy as np


def length_and_mask(x):
    mask = torch.eq(x, data.PAD_ID)
    length = x.size(1) - torch.sum(mask, dim=1)
    return length, mask

def stable_softmax(x, mask=None):
    """
    x: [batch_size, seq_len, out_size] mask: the same shape with x
    or 
    x: [batch_size, out_size] mask: the same shape with x
    """
    orig_size = x.size()
    out_size = orig_size[-1]
    if mask is not None:
        assert x.size() == mask.size()
        mask = mask.contiguous().view(-1, out_size)
        x = x.view(-1, out_size) * (1.0 - mask.float())
    else:
        x = x.view(-1, out_size)
    max_by_row = torch.max(x, dim=1, keepdim=True)[0]
    if mask is not None:
        numerator = torch.exp(x-max_by_row)*(1.0 - mask.float()) 
    else:
        numerator = torch.exp(x-max_by_row)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    softmax = numerator/denominator
    return softmax.view(orig_size)

def save_model(save_dir, model, min_dev_loss):
    save_model_path = os.path.join(save_dir, 'saved_models')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    path = os.path.join(save_model_path, "dev_loss_{}.pth".format(str(min_dev_loss)))
    torch.save(model, path)
    print(f"A model is saved successfully as {path}!")

def save_pred(save_dir, pred_dict, test_acc):
    save_pred_path = os.path.join(save_dir, 'saved_preds')
    if not os.path.exists(save_pred_path):
        os.mkdir(save_pred_path)
    path = os.path.join(save_pred_path, "test_acc_{}.pkl".format(str(test_acc)))
    pickle.dump(pred_dict, open(path, "wb"))
    print(f"A prediction is saved successfully as {path}!")

def save_hps(save_dir, hps):
    path = os.path.join(save_dir, 'hps.pkl')
    pickle.dump(hps, open(path, 'wb'))
    print(f"A config is saved successfully as {path}!")

def load_model(args):
    path = f"saved_models/dev_loss_{args.load_model_name}.pkl"
    try:
        model = pickle.load(open(os.path.join(args.exp_dir, path), "rb"))
        print(f"Model in {path} loaded successfully!")
        return model
    except:
        print(f"No available model such as {path}.")
        exit()

def main():
    vec = np.array([12345, 67890, 99999999])
    stable_softmax(torch.tensor(vec, dtype=torch.float))#, mask=torch.tensor([0,0,0]))

if __name__ == '__main__':
    main()
