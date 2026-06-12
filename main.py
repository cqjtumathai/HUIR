import torch, pickle, time, os
import numpy as np
from torch import nn
from param import args
from DataHander import DataHandler
from models.model import SDNet, GCNModel 
from utils import load_model, save_model, fix_random_seed_as
from tqdm import tqdm
from models import diffusion_process as dp
from Utils.Utils import *
import logging
import sys
import torch.nn.functional as F


class Coach:
    def __init__(self, handler):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.handler = handler
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user, self.n_item = self.handler.n_user, self.handler.n_item
        self.uiGraph = self.handler.ui_graph.to(self.device)
        self.uuGraph = self.handler.uu_graph.to(self.device)

        self.GCNModel = GCNModel(args, self.n_user, self.n_item).to(self.device)
        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.DiffProcess = dp.DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
                                               args.steps, self.device).to(self.device)
        self.optimizer1 = torch.optim.Adam([{'params': self.GCNModel.parameters(), 'weight_decay': 0}], lr=args.lr)
        self.optimizer2 = torch.optim.Adam([{'params': self.SDNet.parameters(), 'weight_decay': 0}], lr=args.difflr)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=args.decay_step, gamma=args.decay)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=args.decay_step, gamma=args.decay)
        self.train_loss = []
        self.his_recall = []
        self.his_ndcg = []

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def SRPC(self, X, adj, user_idx):
        SIGMA = 1e-10
        f = lambda x: torch.exp(x)
        refl_sim = f(self.sim(X, X))
        refl_sim_1 = f(self.sim(X, X))[user_idx]
        between_sim = (adj * refl_sim)[user_idx]
        pos_val = between_sim.sum(1)
        neg_val = refl_sim_1.sum(1) - pos_val
        return -torch.log((pos_val + SIGMA) / (neg_val + SIGMA)).mean()



    def ISEC(self, graph_a, u_s, n, m, user_idx, item_idx):
        A = torch.zeros(n, m)
        src, tar = graph_a.edges()
        mask = src < n
        A_u = src[mask]
        A_i = tar[mask] - n
        A[A_u, A_i] = 1
        A = A.cpu()[user_idx.cpu(), :].to(self.device)
        a = torch.mm(A, A.T)
        b = torch.mm(u_s, u_s.T)
        c = torch.sigmoid(b)
        FP = torch.norm(a - c, p=2)
        return FP

    def train(self):
        args = self.args
        self.save_history = True
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.dataset + '/'
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        best_recall = [0] * len(args.topk_list)
        best_ndcg = [0] * len(args.topk_list)
        best_epoch = 0
        wait = 0
        start_time = time.time()
        for self.epoch in range(1, args.n_epoch + 1):
            epoch_losses = self.train_one_epoch()
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(self.epoch, time.time() - start_time,
                                                                           epoch_losses), flush=True)
            if self.epoch % 5 == 0:
                recall, ndcg = self.test(self.testloader)
                self.his_recall.append(recall)
                self.his_ndcg.append(ndcg)
                for i in range(len(args.topk_list)):
                    if recall[i] > best_recall[i]:
                        best_recall[i] = recall[i]
                        best_epoch = self.epoch
                        wait = 0
                    else:
                        wait += 1
                    if ndcg[i] > best_ndcg[i]:
                        best_ndcg[i] = ndcg[i]
                        best_epoch = self.epoch
                        wait = 0
                    else:
                        wait += 1
                logger.info('+ epoch {} tested, elapsed {:.2f}s'.format(self.epoch, time.time() - start_time))
                for i, topk in enumerate(args.topk_list):
                    logger.info(f'Recall@{topk}: {recall[i]:.4f}, NDCG@{topk}: {ndcg[i]:.4f}')
                if args.model_dir:
                    for i, topk in enumerate(args.topk_list):
                        if recall[i] == best_recall[i] or ndcg[i] == best_ndcg[i]:
                            desc = args.save_name
                            perf = f'N_{ndcg[i]:.4f}/R_{recall[i]:.4f}'
                            fname = f'{args.desc}_{desc}_{perf}_topk{topk}.pth'
                            save_model(self.GCNModel, self.SDNet, os.path.join(args.model_dir, fname), self.optimizer1,
                                       self.optimizer2)
            if self.save_history:
                self.saveHistory()
            if wait >= args.patience:
                print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                break
        print('Best Recall and NDCG:')
        for i, topk in enumerate(args.topk_list):
            print(f'Best Recall@{topk}: {best_recall[i]:.6f}, Best NDCG@{topk}: {best_ndcg[i]:.6f}')

    def train_one_epoch(self):
        self.SDNet.train()
        self.GCNModel.train()
        dataloader = self.train_loader
        epoch_losses = [0] * 3
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)
        since = time.time()
        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long().cuda()
            pos_idx = pos_idx.long().cuda()
            neg_idx = neg_idx.long().cuda()
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]
            user = uEmbeds[user_idx]
            pos = iEmbeds[pos_idx]
            neg = iEmbeds[neg_idx]
            uu_terms = self.DiffProcess.caculate_losses(self.SDNet, uuEmbeds[user_idx], args.reweight)
            uuelbo = uu_terms["loss"].mean()
            u_social = uu_terms["pred_xstart"]
            ISEC_loss = self.ISEC(self.uiGraph, u_social, self.n_user, self.n_item, user_idx, pos_idx)
            user = user + uu_terms["pred_xstart"]
            diffloss = uuelbo
            adj_label = torch.zeros(self.n_user, self.n_user).to(self.device)
            src, tar = self.uuGraph.edges()[0], self.uuGraph.edges()[1]
            adj_label[src, tar] = 1
            adj_label[tar, src] = 1
            index = torch.arange(adj_label.shape[0]).to(self.device)
            adj_label[index, index] = 1
            SRPCloss = self.SRPC(uEmbeds, adj_label, user_idx)
            scoreDiff = pairPredict(user, pos, neg)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch_size
            regLoss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * args.reg) / args.batch_size
            loss = bprLoss + regLoss
            losses = [bprLoss.item(), regLoss.item()]
            loss = diffloss + args.bprloss * loss + args.SRPCloss * SRPCloss + args.ISECloss *ISEC_loss
            losses.append(diffloss.item())
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]
        if self.scheduler1 is not None:
            self.scheduler1.step()
            self.scheduler2.step()
        epoch_losses = [sum(epoch_losses)] + epoch_losses
        time_elapsed = time.time() - since
        print('Training complete in {:.4f}s'.format(
            time_elapsed))
        return epoch_losses

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = [0] * len(args.topk_list)
        allNdcg = [0] * len(args.topk_list)
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            for j, topk in enumerate(args.topk_list):
                maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
                recall = dcg = 0
                for val in temTstLocs:
                    if val in temTopLocs[:topk]:
                        recall += 1
                        dcg += np.reciprocal(np.log2(temTopLocs[:topk].index(val) + 2))
                recall = recall / tstNum
                ndcg = dcg / maxDcg if maxDcg > 0 else 0
                allRecall[j] += recall
                allNdcg[j] += ndcg
        return allRecall, allNdcg

    def test(self, dataloader):
        self.SDNet.eval()
        self.GCNModel.eval()
        Recall = [0] * len(args.topk_list)
        NDCG = [0] * len(args.topk_list)
        num = dataloader.dataset.__len__()
        since = time.time()
        with torch.no_grad():
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trnMask = batch
                user_idx = user_idx.long().cuda()
                trnMask = trnMask.cuda()
                uEmbeds = uiEmbeds[:self.n_user]
                iEmbeds = uiEmbeds[self.n_user:]
                user = uEmbeds[user_idx]
                uuemb = uuEmbeds[user_idx]
                user_predict = self.DiffProcess.p_sample(self.SDNet, uuemb, args.sampling_steps, args.sampling_noise)
                user = user + user_predict
                allPreds = torch.mm(user, torch.transpose(iEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                _, topLocs = torch.topk(allPreds, max(args.topk_list))
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                for j in range(len(args.topk_list)):
                    Recall[j] += recall[j]
                    NDCG[j] += ndcg[j]
            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(time_elapsed))
            Recall = [r / num for r in Recall]
            NDCG = [n / num for n in NDCG]
        return Recall, NDCG

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        for i, topk in enumerate(args.topk_list):
            history[f'Recall@{topk}'] = [recall_list[i] for recall_list in self.his_recall]
            history[f'NDCG@{topk}'] = [ndcg_list[i] for ndcg_list in self.his_ndcg]
        ModelName = "SDR"
        desc = args.save_name
        perf = ''
        fname = f'{args.desc}_{desc}_{perf}.his'
        with open('./History/' + args.dataset + '/' + fname, 'wb') as fs:
            pickle.dump(history, fs)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    fix_random_seed_as(args.seed)
    handler = DataHandler()
    handler.LoadData()
    app = Coach(handler)
    app.train()
