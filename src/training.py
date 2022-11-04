import os
from src.datasets import ABSADataset
from dgl.dataloading import GraphDataLoader
import torch
from sklearn import metrics
from src.trainer import DLGMtrainer


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def training(config):
    train_set = ABSADataset(config, split_name='train')
    valid_set = ABSADataset(config, split_name='valid')
    test_set = ABSADataset(config, split_name='test')
    train_loader = GraphDataLoader(train_set, batch_size=config['batch_size'], drop_last=False, shuffle=True)
    valid_loader = GraphDataLoader(valid_set, batch_size=config['batch_size'], drop_last=False, shuffle=True)
    test_loader = GraphDataLoader(test_set, batch_size=config['batch_size'], drop_last=False, shuffle=True)
    print('number of training samples: {}'.format(len(train_set)))
    print('number of validation samples: {}'.format(len(valid_set)))
    print('number of testing samples: {}'.format(len(test_set)))

    save_model_path = os.path.join(os.path.abspath('..'), 'Amax_DLGM', 'checkpoint')

    trainer = DLGMtrainer(config)

    best_model = save_model_path + '/{}_best_model.pt'.format(config['dataset'])

    train_acc_history, train_loss_history = [0.0], [0.0]
    val_acc_history, val_loss_history, val_f1_score_history = [0.0], [0.0], [0.0]

    for epoch in range(config['epochs']):
        print("Epoch {}".format(epoch + 1) + '-' * 60)
        train_loss, train_acc, train_step = 0.0, 0.0, 0
        for i, batch in enumerate(train_loader):
            loss, acc = trainer.update(batch)
            train_loss += loss
            train_acc += acc
            train_step += 1
            if train_step % config['log_step'] == 0:
                print(
                    "{}/{} train_loss: {:.6f}, train_acc: {:.6f}".format(
                        i + 1, len(train_loader), train_loss / train_step, train_acc / train_step
                    )
                )
        val_loss, val_acc, val_f1 = evaluate(trainer, valid_loader)
        print(
            "End of {} train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score: {:.4f}".format(
                epoch + 1, train_loss / train_step, train_acc / train_step, val_loss, val_acc, val_f1
            )
        )

        train_acc_history.append(train_acc / train_step)
        train_loss_history.append(train_loss / train_step)
        val_loss_history.append(val_loss)

        if epoch + 1 == 1 or float(val_acc) > max(val_acc_history):
            exit()
            torch.save(trainer, best_model)
            print("new best model saved.")

        val_acc_history.append(float(val_acc))
        val_f1_score_history.append(val_f1)
    print("Training ended with {} epochs.".format(config['epochs']))

    print("Loading best checkpoint from ", best_model)
    trainer = torch.load(best_model)
    test_loss, test_acc, test_f1 = evaluate(trainer, test_loader)
    print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(test_loss, test_acc, test_f1))


def evaluate(model, data_loader):
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    for i, batch in enumerate(data_loader):
        loss, acc, pred, label, _ = model.predict(batch)
        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average="macro")
    return val_loss / val_step, val_acc / val_step, f1_score