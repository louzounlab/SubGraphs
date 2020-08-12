import math
import time
import os
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import nni
import logging
import networkx as nx
from loggers import EmptyLogger, CSVLogger, PrintLogger, FileLogger, multi_logger
from model import GCN, GatNet
from pre_peocess import build_2k_vectors
import pickle

CUDA_Device = 1

class ModelRunner:
    def __init__(self, conf, logger, data_logger=None, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self.bar = 0.5
        self._lr = conf["lr"]
        self._is_nni = is_nni
        # choosing GPU device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device != "cpu":
            with torch.cuda.device("cuda:{}".format(CUDA_Device)):
                torch.cuda.empty_cache()
            if not self._is_nni:
                self._device = torch.device("cuda:{}".format(CUDA_Device))
        self._loss = self._sub_graph_ce_loss
        self._ce_loss = torch.nn.CrossEntropyLoss(reduction="mean").to(self._device)


    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger

    def _sub_graph_ce_loss(self, calcs, beta=None, gamma=None):
        # if beta is None:
        #  beta = 1 / len(calcs["f_ns_out"])  if len(calcs["f_ns_out"])!=0 else 0
        #  gamma = 1 / len(calcs["s_ns_out"])  if len(calcs["s_ns_out"])!=0 else 0
        #todo check dimensions of central nodes torch
        cn_loss = self._ce_loss(calcs["cn_out"], calcs["cn_label"])
        f_ns_loss = self._ce_loss(calcs["f_ns_out"], calcs["f_ns_labels"]) *(beta) if len(calcs["f_ns_out"])!=0 else 0
        s_ns_loss =  self._ce_loss(calcs["s_ns_out"], calcs["s_ns_labels"]) * (gamma) if len(calcs["s_ns_out"])!=0 else 0
        return cn_loss+f_ns_loss+s_ns_loss


    def _get_model(self):
        model = GCN(in_features=self._conf["in_features"],
                    hid_features=self._conf["hid_features"], out_features= self._conf["out_features"],
                    activation=self._conf["activation"], dropout= self._conf["dropout"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])
        ##checged : added "feature_matrices"
        return {"model": model, "optimizer": opt,
                # "training_mats": self._conf["training_mat"],
                # "training_labels": self._conf["training_labels"],
                # "test_mats": self._conf["test_mat"],
                # "test_labels": self._conf["test_labels"],
                "cut": self._conf["cut"],"beta": self._conf["beta"],"gamma": self._conf["gamma"],
                "labels": self._conf["labels"], "X": self._conf["X"], "ds_name": self._conf["ds_name"],
                "train_ind":  self._conf["train_ind"], "test_ind":  self._conf["test_ind"],
                "adj_matrices": self._conf["adj_matrices"]
                }



    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):

        if self._is_nni:
            verbose = 0
        model = self._get_model()
        ##
        loss_train, acc_train, intermediate_acc_test, losses_train, accs_train,  accs_cn_train, accs_f_train, accs_s_train, test_results = self.train(
            self._conf["epochs"],
            model=model,
            verbose=verbose)
        ##
        # Testing . ## result is only the last one! do not use. same as 7 in last
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0, print_to_file=True)
        test_results.append(result)
        if self._is_nni:
            self._logger.debug('Final loss train: %3.4f' % loss_train)
            self._logger.debug('Final accuracy train: %3.4f' % acc_train)
            final_results = result["acc"]
            self._logger.debug('Final accuracy test: %3.4f' % final_results)
            # _nni.report_final_result(test_auc)

        if verbose != 0:
            names = ""
            vals = ()
            for name, val in result.items():
                names = names + name + ": %3.4f  "
                vals = vals + tuple([val])
                self._data_logger.info(name, val)
        parameters = {"temporal_pen": self._conf["temporal_pen"], "lr": self._conf["lr"],
                      "weight_decay": self._conf["weight_decay"],
                      "dropout": self._conf["dropout"], "optimizer": self._conf["optim_name"]}
        return loss_train, acc_train, intermediate_acc_test, result, losses_train, accs_train, accs_cn_train, accs_f_train, accs_s_train, test_results, parameters




    def train(self, epochs, model=None, verbose=2):
        loss_train = 0.
        acc_train = 0.
        losses_train = []
        accs_train = []
        accs_cn_train = []
        accs_f_train = []
        accs_s_train = []


        test_results = []
        intermediate_test_acc = []
        for epoch in range(epochs):
            loss_train, acc_train, acc_train_cn , acc_train_f, acc_train_s= self._train(epoch, model, verbose)
            ##
            losses_train.append(loss_train)
            accs_train.append(acc_train)
            accs_cn_train.append(acc_train_cn)
            #if acc_train_f!=0:
            accs_f_train.append(acc_train_f)
        # if acc_train_s!=0:
            accs_s_train.append(acc_train_s)
            ##
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                test_results.append(test_res)
                if self._is_nni:
                    test_acc = test_res["acc"]
                    intermediate_test_acc.append(test_acc)

        return loss_train, acc_train, intermediate_test_acc, losses_train, \
                accs_train, accs_cn_train, accs_f_train, accs_s_train, test_results

    def calculate_labels_outputs(self,node,  outputs , labels, indices, ego_graph):
        f_neighbors = list(ego_graph.neighbors(node))
        s_neighbors = []
        for f_neighbor in f_neighbors:
            for s_neighbor in ego_graph.neighbors(f_neighbor):
                if s_neighbor not in f_neighbors and s_neighbor != node and s_neighbor not in s_neighbors:
                    s_neighbors += [s_neighbor]
        cn_out= outputs[[list(ego_graph.nodes).index(node)]]
        cn_label = labels[[node]] ##todo [node]

        f_ns_out = outputs[[list(ego_graph.nodes).index(f_n) for f_n in f_neighbors if f_n in indices]]
        f_ns_labels = labels[[f_n for f_n in f_neighbors if f_n in indices]]

        s_ns_out = outputs[[list(ego_graph.nodes).index(s_n) for s_n in s_neighbors if s_n in indices]]
        s_ns_labels = labels[[s_n for s_n in s_neighbors if s_n in indices]]
        return { "cn_out": cn_out, "cn_label":  cn_label, "f_ns_out": f_ns_out, "f_ns_labels": f_ns_labels,  "s_ns_out": s_ns_out, "s_ns_labels": s_ns_labels }


    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        cut = model["cut"]

        train_indices = model["train_ind"]
        model["labels"] = model["labels"].to(self._device)
        labels = model["labels"]
        beta = model["beta"]
        gamma = model["gamma"]
        model_.train()
        optimizer.zero_grad()

        loss_train = 0.
        acc_train = 0
        acc_train_cn, acc_train_f, acc_train_s = 0,0,0
        f_nones = 0; s_nones = 0

        # create subgraphs only for partial, but use labels of all.
        partial_train_indices = train_indices[0:int(cut*len(train_indices))]
        for node in partial_train_indices: #this may be in batches for big graphs todo
            adj = model["adj_matrices"][node]
            X_t = model["X"][list(adj.nodes)].to(device=self._device)
            output = model_(X_t, nx.adjacency_matrix(adj).tocoo())

            calcs = self.calculate_labels_outputs( node, output, labels, train_indices, adj)

            loss_train += self._loss(calcs, beta, gamma)

            acc, acc_cn, acc_f, acc_s = self.accuracy(calcs)
            acc_train_cn+= acc_cn
            if acc_f!=None:
                acc_train_f += acc_f
            else:
                f_nones+=1
            if acc_s!=None:
                acc_train_s+=acc_s
            else:
                s_nones+=1
            acc_train += acc



        loss_train /= len(partial_train_indices)
        acc_train_cn /= len(partial_train_indices)
        if len(partial_train_indices)-f_nones !=0:
            acc_train_f /= (len(partial_train_indices)-f_nones)
        else:
            acc_train_f = np.nan
        if len(partial_train_indices)-s_nones !=0:
            acc_train_s  /= (len(partial_train_indices)-s_nones)
        else:
            acc_train_s = np.nan
        acc_train/= len(partial_train_indices)
        #print("Train Acc on cn", acc_train_cn / 1, "Acc first nodes", acc_train_f, "Acc second nodes", acc_train_s)

        loss_train.backward()
        optimizer.step()

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'ce_loss_train: {:.4f} '.format(loss_train.data.item()) +

                               'acc_train: {:.4f} '.format(acc_train))
        return loss_train, acc_train, acc_train_cn , acc_train_f, acc_train_s



    @staticmethod
    def accuracy(calcs):
        # return {"cn_out": cn_out, "cn_label": cn_label, "f_ns_out": f_ns_out, "f_ns_labels": f_ns_labels,
        #         "s_ns_out": s_ns_out, "s_ns_labels": s_ns_labels}
        acc = 0
        acc_cn, acc_f, acc_s = 0,0,0
        for idx, sample in enumerate(calcs["f_ns_out"]):
            if torch.argmax(sample) == calcs["f_ns_labels"][idx]:
                acc+=1
                acc_f+=1
        for idx, sample in enumerate(calcs["s_ns_out"]):
            if torch.argmax(sample) == calcs["s_ns_labels"][idx]:
                acc+=1
                acc_s+=1
        if torch.argmax(calcs["cn_out"]) == calcs["cn_label"]:
            acc+=1
            acc_cn+=1
        size_labeld = len(calcs["cn_out"])+len(calcs["s_ns_out"])+len(calcs["f_ns_out"])
        #print(acc_cn, acc_f,acc_s)
        acc_f = acc_f/len(calcs["f_ns_out"]) if len(calcs["f_ns_out"])!=0 else None
        acc_s = acc_s / len(calcs["s_ns_out"]) if len(calcs["s_ns_out"]) != 0 else None
        #print("Acc on cn", acc_cn/1, "Acc first nodes", acc_f,  "Acc second nodes",acc_s)
        #return acc/size_labeld  # for all with no change between first and seconds
        return acc/size_labeld, acc_cn/1, acc_f, acc_s






    def test(self, model=None, verbose=2, print_to_file=False):
        model_ = model["model"]
        test_indices = model["test_ind"]
        labels = model["labels"]
        beta = model["beta"]
        gamma = model["gamma"]
        model_.eval()

        test_loss = 0
        test_acc = 0
        acc_test_cn, acc_test_f, acc_test_s = 0, 0, 0
        f_nones= 0; s_nones= 0

        partial_rand_test_indices = np.random.choice(len(test_indices), round(0.05*len(test_indices)) , replace=False)
        #partial_rand_test_indices = test_indices
        #partial_test_indices = test_indices[0:int(1 * len(test_indices))] ## 1 is all
        for node in partial_rand_test_indices: #this may be in batches for big graphs todo
            #adj is the ego graph (that will be converted into adj matrix and coo).
            adj = model["adj_matrices"][node]
            import random
            random.shuffle(adj.nodes)
            X_t = model["X"][list(adj.nodes)].to(device=self._device)
            print(X_t[0])
            random.shuffle(X_t)
            print(X_t[0],"after")
            #todo this may be given as another param, to avoid using cpu calculations here
            output = model_(X_t, nx.adjacency_matrix(adj).tocoo())

            calcs = self.calculate_labels_outputs( node, output, labels, test_indices, adj)

            test_loss += self._loss(calcs, beta, gamma)

            #test_acc += self.accuracy(calcs)
            acc, acc_cn, acc_f, acc_s = self.accuracy(calcs)
            acc_test_cn += acc_cn
            if acc_f!=None:
                acc_test_f += acc_f
            else:
                f_nones +=1
            if acc_s != None:
                acc_test_s += acc_s
            else:
                s_nones +=1
            test_acc += acc



        test_loss /= len(partial_rand_test_indices)
        test_acc /= len(partial_rand_test_indices)
        acc_test_cn /= len(partial_rand_test_indices); acc_test_f /= (len(partial_rand_test_indices)-f_nones); acc_test_s /= (len(partial_rand_test_indices)-s_nones)
        #print("Test Acc on cn", acc_test_cn/1, "Acc first nodes", acc_test_f,  "Acc second nodes",acc_test_s)


        if verbose != 0:
            self._logger.info("Test: ce_loss= {:.4f} ".format(test_loss.data.item()) + "acc= {:.4f}".format(test_acc))
        #result = {"loss": loss_test.data.item(), "acc": acc_test, "tempo_loss": tempo_loss.data.item()}


        result = {"loss": test_loss, "acc": test_acc, "acc_cn": acc_test_cn, "acc_f":acc_test_f, "acc_s":acc_test_s}
        return result



def plot_graphs(train_loss_mean, train_acc_mean,train_cn_acc_mean,train_f_acc_mean, train_s_acc_mean, test_loss_mean, test_acc_mean,
                test_cn_acc_mean,test_f_acc_mean,test_s_acc_mean, parameters, plots_data):
    # info[4] is list of train losses 1 .  list[5] is list of acc train.
    #info [6] is list of dictionaries, each dictionary is for epoch, each one contains "loss" - first loss,"acc"- acc,  "tempo_loss" - tempo loss
    #info[7] is the temporal_oen

    regulariztion = str(round(parameters["weight_decay"],3))
    lr = str(round(parameters["lr"],3))
    optimizer = str(parameters["optimizer"])
    dropout = str(round(parameters["dropout"],2))

    cut = plots_data["cut"]*100
    ds_name = plots_data["ds_name"]



    #Train

    # Share a X axis with each column of subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    plt.suptitle("DataSet: " + ds_name
                    + ", final_train_accuracies_mean: " + str(round(plots_data["final_train_accuracies_mean"],2)) + ", final_train_accuracies_ste: " + str(round(plots_data["final_train_accuracies_ste"],2))
                    + "\nfinal_test_accuracies_mean: " + str(round(plots_data["final_test_accuracies_mean"],2)) + ", final_test_accuracies_ste: " + str(round(plots_data["final_test_accuracies_ste"],2))
                 + "\nlr="+lr+" reg= "+regulariztion+ ", dropout= "+dropout+", opt= "+optimizer, fontsize=12, y=0.99)

    epoch = [e for e in range(1, len(train_loss_mean)+1)]
    axes[0, 0].set_title('Loss train')
    axes[0, 0].set_xlabel("epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].plot(epoch, train_loss_mean)

    axes[0, 1].set_title('Accuracy train')
    axes[0, 1].set_xlabel("epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].plot(epoch, train_acc_mean)


    axes[0, 2].set_title('Accuracy layers Train')
    axes[0, 2].set_xlabel("epochs")
    axes[0, 2].set_ylabel("Accuracies")
    axes[0, 2].plot(epoch, train_cn_acc_mean, label='CentralNode')
    axes[0, 2].plot(epoch, train_f_acc_mean, label='FirstNeighbors')
    axes[0, 2].plot(epoch, train_s_acc_mean, label='SecondNeighbors')
    axes[0, 2].legend(loc='best')


    #test

    epoch = [e for e in range(1, len(test_loss_mean)+1)]

    axes[1, 0].set_title('Loss test')
    axes[1, 0].set_xlabel("epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].plot(epoch, test_loss_mean)


    axes[1, 1].set_title('Accuracy test')
    axes[1, 1].set_xlabel("epochs")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].plot(epoch, test_acc_mean)


    axes[1, 2].set_title('Accuracy layers Test')
    axes[1, 2].set_xlabel("epochs")
    axes[1, 2].set_ylabel("Accuracies")
    axes[1, 2].plot(epoch, test_cn_acc_mean, label='CentralNode')
    axes[1, 2].plot(epoch, test_f_acc_mean, label='FirstNeighbors')
    axes[1, 2].plot(epoch, test_s_acc_mean, label='SecondNeighbors')

    axes[1, 2].legend(loc='best')

    fig.tight_layout()
    plt.subplots_adjust(top=0.85)

    # fig.delaxes(axes[1,0])
    plt.savefig("figures/"+plots_data["ds_name"]+"_.png")

    plt.clf()
    #plt.show()




def execute_runner(runners, plots_data, is_nni=False):
    train_losses = []
    train_accuracies = []
    train_cn_accuracies = []
    train_f_accuracies = []
    train_s_accuracies = []
    test_intermediate_results = []
    test_losses = []
    test_accuracies = []
    test_cn_accuracies = []
    test_f_accuracies = []
    test_s_accuracies = []
    results = []
    last= runners[-1]
    for i in range(len(runners)):
    #for idx_r, runner in enumerate(runners):
        with torch.cuda.device("cuda:{}".format(CUDA_Device)):
            torch.cuda.empty_cache()
            time.sleep(1)
        print("trial number",i)
        result_one_iteration = runners[0].run(verbose=2)
        train_losses.append(result_one_iteration[0])
        train_accuracies.append(result_one_iteration[1])
        test_intermediate_results.append(result_one_iteration[2])
        test_losses.append(result_one_iteration[3]["loss"])
        test_accuracies.append(result_one_iteration[3]["acc"])
        results.append(result_one_iteration)
        #todo check if can be deleted (from first check - not changing)
        if len(runners) >1:
            runners=runners[1:]
        print("len runners", len(runners))

    # for printing results on graphs. for other uses - the last result is the one should be used.
    size = len(results)
    #train_loss_mean = torch.stack([torch.tensor([results[j][4][i] for i in range(len(results[j][4]))]) for j in range(size)]).mean(axis=0)
    train_loss_mean = np.mean([ [results[j][4][i].item() for i in range(len(results[j][4]))] for j in range(size) ], axis=0)
    #train_acc_mean = torch.stack([ torch.tensor([results[j][5][i] for i in range(len(results[j][5]))]) for j in range(size) ]).mean(axis=0)
    train_acc_mean = np.mean([ [results[j][5][i] for i in range(len(results[j][5]))] for j in range(size) ], axis=0)
    train_cn_acc_mean = np.mean([[results[j][6][i] for i in range(len(results[j][6]))] for j in range(size)], axis=0)
    train_f_acc_mean = np.nanmean([[results[j][7][i] for i in range(len(results[j][7]))] for j in range(size)], axis=0)
    train_s_acc_mean = np.nanmean([[results[j][8][i] for i in range(len(results[j][8]))] for j in range(size)], axis=0)
    #test_loss_mean = torch.stack([ torch.tensor([results[j][6][i]["loss"] for i in range(len(results[j][6]))]) for j in range(size) ]).mean(axis=0)
    test_loss_mean = np.mean([ [results[j][9][i]["loss"].item() for i in range(len(results[j][9]))] for j in range(size) ], axis=0)
    #test_acc_mean = torch.stack([ torch.tensor([torch.tensor(results[j][6][i]["acc"]) for i in range(len(results[j][6]))]) for j in range(size) ])
    test_acc_mean = np.mean([ [results[j][9][i]["acc"] for i in range(len(results[j][9]))] for j in range(size) ], axis=0 )
    test_cn_acc_mean = np.mean([[results[j][9][i]["acc_cn"] for i in range(len(results[j][9]))] for j in range(size)], axis=0)
    test_f_acc_mean = np.mean([[results[j][9][i]["acc_f"] for i in range(len(results[j][9]))] for j in range(size)], axis=0)
    test_s_acc_mean = np.mean([[results[j][9][i]["acc_s"] for i in range(len(results[j][9]))] for j in range(size)], axis=0) #todo take care of None here too?

    final_train_accuracies_mean = np.mean(train_accuracies)
    final_train_accuracies_ste = np.std(train_accuracies) / math.sqrt(len(runners))
    final_test_accuracies_mean = np.mean(test_accuracies)
    final_test_accuracies_ste = np.std(test_accuracies) / math.sqrt(len(runners))

    plots_data["final_train_accuracies_mean"] = final_train_accuracies_mean
    plots_data["final_train_accuracies_ste"] = final_train_accuracies_ste
    plots_data["final_test_accuracies_mean"] = final_test_accuracies_mean
    plots_data["final_test_accuracies_ste"] = final_test_accuracies_ste

    #plot to graphs
    plot_graphs(train_loss_mean, train_acc_mean,train_cn_acc_mean,train_f_acc_mean, train_s_acc_mean, test_loss_mean, test_acc_mean,
                test_cn_acc_mean,test_f_acc_mean,test_s_acc_mean, results[0][10], plots_data)

    if is_nni:
        mean_intermediate_res = np.mean(test_intermediate_results, axis=0)
        for i in mean_intermediate_res:
            nni.report_intermediate_result(i)
        nni.report_final_result(np.mean(test_accuracies))



    # T takes the final of each iteration and for them mkes mean and std
    last.logger.info("*" * 15 + "Final accuracy train: %3.4f" % final_train_accuracies_mean)
    last.logger.info("*" * 15 + "Std accuracy train: %3.4f" % final_train_accuracies_ste)
    last.logger.info("*" * 15 + "Final accuracy test: %3.4f" % final_test_accuracies_mean)
    last.logger.info("*" * 15 + "Std accuracy test: %3.4f" % final_test_accuracies_ste)
    last.logger.info("Finished")
    return



def build_model(rand_test_indices, train_indices, labels ,adjacency_matrices,X,in_features,
                hid_features,out_features,ds_name, cut, activation, optimizer, epochs, dropout, lr, l2_pen, temporal_pen,
                beta, gamma, dumping_name, is_nni=False):
    optim_name="SGD"
    if optimizer==optim.Adam:
        optim_name = "Adam"
    conf = {"in_features":in_features, "hid_features": hid_features, "out_features":out_features,"ds_name":ds_name, "cut": cut,
            "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
            "temporal_pen": temporal_pen, "beta": beta, "gamma": gamma,
            #"training_mat": training_data, "training_labels": training_labels,
            # "test_mat": test_data, "test_labels": test_labels,
            "train_ind": train_indices, "test_ind": rand_test_indices, "labels":labels, "X":X,
            "adj_matrices": adjacency_matrices,
            "optimizer": optimizer, "epochs": epochs, "activation": activation,"optim_name":optim_name}

    products_path = os.path.join(os.getcwd(), "logs", dumping_name, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_%s" % dumping_name, path=products_path, level=logging.INFO)], name=None)

    data_logger = CSVLogger("results_%s" % dumping_name, path=products_path)
    data_logger.info("model_name", "loss", "acc")

    # ##
    # logger.info('STARTING with cut= {:.3f} '.format(cut*100) + ' lr= {:.4f} '.format(lr) + ' dropout= {:.4f} '.format(dropout)+ ' regulariztion_l2_pen= {:.4f} '.format(l2_pen)
    #             + ' temporal_pen= {:.10f} '.format(temporal_pen)+ ' beta= {:.5f} '.format(beta)+ ' gamma= {:.5f} '.format(gamma)+ ' optimizer= %s ' %optim_name)
    # logger.debug('STARTING with lr=  {:.4f} '.format(lr) + ' dropout= {:.4f} '.format(dropout) + ' regulariztion_l2_pen= {:.4f} '.format(l2_pen)
    #     + ' temporal_pen= {:.10f} '.format(temporal_pen) +' beta= {:.5f} '.format(beta)+ ' gamma= {:.5f} '.format(gamma)+  ' optimizer= %s ' %optim_name)
    # ##

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, is_nni=is_nni)
    return runner



def main_gcn(adj_matrices, X, labels,in_features, hid_features, out_features, ds_name, cut,
             optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, temporal_pen=1e-6, beta=1/4, gamma = 1/16,
             trials=1, dumping_name='', is_nni=False):
    plot_data = {"ds_name": ds_name, "cut": cut}
    runners = []
    #np.random.seed(2)

    #print("epochs", epochs,"l2_pen", l2_pen,"dropout", dropout,"dropout", cut,"cut", dropout)
    for it in range(trials):
        num_classes = out_features
        rand_test_indices = np.random.choice(len(labels), len(labels)-(20*num_classes), replace=False) #
        train_indices = np.delete(np.arange(len(labels)), rand_test_indices)
        #train_indices = train_indices[0:int(cut*len(train_indices))]
        #create x - releveant for 2k only
        # X = build_2k_vectors(ds_name, out_features, train_indices)
        activation = torch.nn.functional.relu

        runner = build_model(rand_test_indices, train_indices, labels, adj_matrices,X, in_features, hid_features,
                             out_features,ds_name,cut, activation, optimizer, epochs, dropout, lr,
                             l2_pen, temporal_pen, beta, gamma, dumping_name, is_nni=is_nni)

        runners.append(runner)

    execute_runner(runners, plot_data, is_nni=is_nni)
    return