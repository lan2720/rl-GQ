import os
import pickle
import torch

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


