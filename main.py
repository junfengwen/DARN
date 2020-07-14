import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim

from model import DarnMLP, DarnConv, MdmnMLP, MdmnConv, MsdaMLP, MsdaConv
from load_data import load_numpy_data, data_loader, multi_data_loader
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name of the dataset: [amazon|digits|office_home].",
                    type=str, choices=['amazon', 'digits', 'office_home'], default="amazon")
parser.add_argument("--method", help="Choose a method: [darn|src|tar|dann|mdmn|msda].",
                    type=str, choices=['darn', 'src', 'tar', 'dann', 'mdmn', 'msda'], default="darn")
parser.add_argument("--result_path", help="Where to save results.",
                    type=str, default="./results")
parser.add_argument("--data_path", help="Where to find the data.",
                    type=str, default="./datasets")
parser.add_argument("--mode", help="Aggregation mode [dynamic|L2]: L2 for DARN, dynamic for MDAN.",
                    type=str, choices=['dynamic', 'L2'], default="L2")
parser.add_argument("--lr", help="Learning rate.",
                    type=float, default=0.5)
parser.add_argument("--mu", help="Hyperparameter of the coefficient for the domain adversarial loss.",
                    type=float, default=1e-2)
parser.add_argument("--gamma", help="Inverse temperature hyperparameter.",
                    type=float, default=1.0)
parser.add_argument("--epoch", help="Number of training epochs.",
                    type=int, default=50)
parser.add_argument("--batch_size", help="Batch size during training.",
                    type=int, default=20)
parser.add_argument("--cuda", help="Which cuda device to use.",
                    type=int, default=0)
parser.add_argument("--seed", help="Random seed.",
                    type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size

result_path = os.path.join(args.result_path,
                           args.name,
                           args.method,
                           args.mode)
if not os.path.exists(result_path):
    os.makedirs(result_path)

logger = utils.get_logger(os.path.join(result_path,
                                       "gamma_%g_seed_%d.log" % (args.gamma,
                                                                 args.seed)))
logger.info("Hyperparameter setting = %s" % args)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

#################### Loading the datasets ####################

time_start = time.time()

data_names, train_insts, train_labels, test_insts, test_labels, configs = load_numpy_data(args.name,
                                                                                          args.data_path,
                                                                                          logger)
configs["mode"] = args.mode
configs['mu'] = args.mu
configs["gamma"] = args.gamma
configs["num_src_domains"] = len(data_names) - 1
num_datasets = len(data_names)

logger.info("Time used to process the %s = %g seconds." % (args.name, time.time() - time_start))
logger.info("-" * 100)

test_results = {}
np_test_results = np.zeros(num_datasets)

#################### Model ####################

if args.method in ["dann", "src", "tar"]:
    # Combine all sources for these methods
    num_src_domains = configs["num_src_domains"] = 1
else:
    num_src_domains = configs["num_src_domains"]

logger.info("Model setting = %s." % configs)

#################### Train ####################

alpha_list = np.zeros([num_datasets, num_src_domains, args.epoch])

for i in range(num_datasets):

    # Build source instances
    source_insts = []
    source_labels = []
    for j in range(num_datasets):
        if j != i:
            source_insts.append(train_insts[j].astype(np.float32))
            source_labels.append(train_labels[j].astype(np.int64))

    # Build target instances
    target_insts = train_insts[i].astype(np.float32)
    target_labels = train_labels[i].astype(np.int64)

    # Model
    if args.method in ["darn", "dann", "src", "tar"]:

        if args.method == "dann" or args.method == "src":
            source_insts = [np.concatenate(source_insts, axis=0)]
            source_labels = [np.concatenate(source_labels)]
        elif args.method == "tar":
            source_insts = [target_insts]
            source_labels = [target_labels]

        if args.method == "src" or args.method == "tar":
            configs["mu"] = 0.

        if args.name in ["amazon", "office_home"]:  # MLP
            model = DarnMLP(configs).to(device)
        elif args.name == "digits":  # ConvNet
            model = DarnConv(configs).to(device)

    elif args.method == "mdmn":

        if args.name in ["amazon", "office_home"]:
            model = MdmnMLP(configs).to(device)
        elif args.name == "digits":
            model = MdmnConv(configs).to(device)

    elif args.method == "msda":

        if args.name in ["amazon", "office_home"]:
            model = MsdaMLP(configs).to(device)
        elif args.name == "digits":
            model = MsdaConv(configs).to(device)

    else:
        raise ValueError("Unknown method")

    if args.method == 'msda':

        opt_G = optim.Adadelta(model.G_params, lr=args.lr)
        opt_C1 = optim.Adadelta(model.C1_params, lr=args.lr)
        opt_C2 = optim.Adadelta(model.C2_params, lr=args.lr)

    else:

        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Training phase
    model.train()
    time_start = time.time()
    for t in range(args.epoch):

        running_loss = 0.0
        train_loader = multi_data_loader(source_insts, source_labels, batch_size)

        for xs, ys in train_loader:

            for j in range(num_src_domains):

                xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)

            ridx = np.random.choice(target_insts.shape[0], batch_size)
            tinputs = target_insts[ridx, :]
            tinputs = torch.tensor(tinputs, requires_grad=False).to(device)

            if args.method != 'msda':
                optimizer.zero_grad()
                loss, alpha = model(xs, ys, tinputs)
                loss.backward()
                optimizer.step()
            else:  # special training step for msda
                loss = utils.msda_train_step(model, xs, ys, tinputs, opt_G, opt_C1, opt_C2)

            running_loss += loss.item()

        if args.method == 'mdmn' or (args.method == 'darn' and args.mode == 'L2'):

            logger.info("Epoch %d, Alpha on %s: %s" % (t, data_names[i], alpha))
            alpha_list[i, :, t] = alpha

        logger.info("Epoch %d, loss = %.6g" % (t, running_loss))

    logger.info("Finish training %s in %.6g seconds" % (data_names[i],
                                                        time.time() - time_start))

    model.eval()

    # Test (use another hold-out target)
    test_loader = data_loader(test_insts[i], test_labels[i], batch_size=1000, shuffle=False)
    test_acc = 0.
    for xt, yt in test_loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
        preds_labels = torch.squeeze(torch.max(model.inference(xt), 1)[1])
        test_acc += torch.sum(preds_labels == yt).item()
    test_acc /= test_insts[i].shape[0]
    logger.info("Test accuracy on %s = %.6g" % (data_names[i], test_acc))
    test_results[data_names[i]] = test_acc
    np_test_results[i] = test_acc

logger.info("All test accuracies: ")
logger.info(test_results)

# Save results to files
test_file = os.path.join(result_path,
                         "gamma_%g_seed_%d_test.txt" % (args.gamma,
                                                        args.seed))
np.savetxt(test_file, np_test_results, fmt='%.6g')

if args.method == 'mdmn' or (args.method == 'darn' and args.mode == 'L2'):
    for i in range(num_datasets):
        alpha_file = os.path.join(result_path,
                                  "gamma_%g_seed_%d_alpha%d.txt" % (args.gamma,
                                                                    args.seed,
                                                                    i))
        np.savetxt(alpha_file, alpha_list[i], fmt='%.6g')

logger.info("Done")
logger.info("*" * 100)
