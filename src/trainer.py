import torch
import numpy as np
from src.model import DLGM
from utils import optim_utils
from utils.loss_function import fina_loss
import torch.nn.functional as F


class DLGMtrainer(object):
    def __init__(self, config):
        self.config = config
        self.model = DLGM(config)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.cuda()
        self.optimizer = optim_utils.get_optimizer(config['optimizer'], self.parameters, config['learning_rate'])

    def load(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            print('model saved to {}'.format(filename))
        except BaseException:
            print('Warning: saving model failed... continuing anyway!!!')

    def update(self, batch):
        graphs, absa_labels = batch
        graphs, absa_labels = graphs.to('cuda'), absa_labels.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        absa_logit = self.model(graphs)
        # loss = fina_loss(graph, absa_logit, absa_labels, self.config)
        loss = F.cross_entropy(absa_logit, absa_labels)
        # print('main_loss: {} --- linguistic_loss: {} --- independence_loss: {}'.format(main_loss, ling_loss, inde_loss))
        # alpha = self.config['alpha']
        # beta = self.config['beta']
        # loss = alpha * main_loss + beta * ling_loss + 0.4 * inde_loss
        corrects = (torch.max(absa_logit, 1)[1].view(absa_labels.size()).data == absa_labels.data).sum()
        acc = 100.0 * np.float(corrects) / absa_labels.size()[0]

        loss.backward()
        self.optimizer.step()
        return loss.data, acc

    def predict(self, batch):
        graphs, absa_labels = batch
        graphs, absa_labels = graphs.to('cuda'), absa_labels.cuda()

        self.model.eval()
        absa_logits = self.model(graphs)
        # loss = fina_loss(graph, absa_logits, absa_labels, self.config)
        loss = F.cross_entropy(absa_logits, absa_labels)
        # alpha = self.config['alpha']
        # beta = self.config['beta']
        # loss = alpha * main_loss + beta * ling_loss + 0.4 * inde_loss
        corrects = (torch.max(absa_logits, 1)[1].view(absa_labels.size()).data == absa_labels.data).sum()
        acc = 100.0 * np.float(corrects) / absa_labels.size()[0]
        predictions = np.argmax(absa_logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(absa_logits, dim=1).data.cpu().numpy().tolist()

        return (
            loss.data,
            acc,
            predictions,
            absa_labels.data.cpu().numpy().tolist(),
            predprob
        )