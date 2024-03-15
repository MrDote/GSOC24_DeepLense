import os

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics import R2Score

import numpy as np


def _onehot_encode(tensor, num_classes, device = None):
    '''
    Encodes the given tensor into one-hot vectors.
    '''
    return torch.eye(num_classes).to(device).index_select(dim=0, index=tensor.to(device))



def _accuracy_calc(predictions: Tensor, labels: Tensor):

    num_eg = labels.shape[0]
    correct_pred = torch.sum(predictions == labels)
    accuracy = correct_pred.item() / num_eg

    return accuracy




# _accuracy_calc(torch.tensor([[-0.3, -0.5, 2], [0.1, 0.2, 0.7]]), torch.tensor([1, 0]))

# r2_score = R2Score()
# r2_score.reset()

# outputs = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.2, 0.7]])
# labels = torch.tensor([2, 1])

# labels = _onehot_encode(labels, 3)

# print(outputs)
# print(labels)

# r2_score.update(outputs, labels)




def train_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, epochs = 1, step_size: int = 5):
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = model.to(device)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.1)


    best_accuracy = 0.
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    r2_score = R2Score(num_outputs=3)

    r2_score.reset()
    train_r2_list = []
    test_r2_list = []


    for epoch in range(epochs):

        print(f"Training and testing for epoch {epoch+1} began with learning rate: {optimizer.param_groups[0]['lr']}")

        batch_loss = 0.
        batch_accuracy = 0.
        batch_count = 0

        model.train()

        for _, (data, labels) in enumerate(train_loader):

            data = data.to(device)
            labels: Tensor = labels.to(device)

            outputs: Tensor = model(data)

            # print("Outputs shape: ", outputs.shape)
            # print("Labels shape: ", labels.shape)

            loss: Tensor = criterion(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                outputs = nn.functional.softmax(outputs, dim=1)

                batch_loss += loss.item()
                batch_accuracy += _accuracy_calc(torch.argmax(outputs, dim=1), labels=labels)

                r2_score.update(outputs.to('cpu'), _onehot_encode(labels, 3, 'cpu'))

                batch_count += 1

        

        avg_batch_loss = batch_loss/batch_count
        epoch_accuracy = batch_accuracy/batch_count

        print(f"Epoch : {epoch+1}, Training Loss : {round(avg_batch_loss, 3)}, Training Accuracy : {round(epoch_accuracy, 3)}")

        train_loss_list.append(avg_batch_loss)
        train_acc_list.append(epoch_accuracy)


        r2 = r2_score.compute()
        train_r2_list.append(r2)
        print(f"Training R2 : {round(train_r2_list[-1].item(), 3)}")
        r2_score.reset()


        with torch.no_grad():
            #* TESTING
            batch_loss = 0
            batch_accuracy = 0
            batch_count = 0


            model.eval()

            for _, (data, labels) in enumerate(test_loader):

                data = data.to(device)
                labels: Tensor = labels.to(device)

                outputs: Tensor = model(data)
                loss: Tensor = criterion(outputs, labels)
                
                outputs = nn.functional.softmax(outputs, dim=1)

                batch_loss += loss.item()
                batch_accuracy += _accuracy_calc(torch.argmax(outputs, dim=1), labels=labels)

                r2_score.update(outputs.to('cpu'), _onehot_encode(labels, 3, 'cpu'))

                batch_count += 1
            

            avg_batch_loss = batch_loss/batch_count
            epoch_accuracy = batch_accuracy/batch_count

            print(f"Epoch : {epoch + 1}, Testing Loss : {round(avg_batch_loss, 3)}, Testing Accuracy : {round(epoch_accuracy, 3)}")

            test_loss_list.append(avg_batch_loss)
            test_acc_list.append(epoch_accuracy)


            r2 = r2_score.compute()
            test_r2_list.append(r2)
            print(f"Testing R2 : {round(test_r2_list[-1].item(), 3)}")
            r2_score.reset()

        
        lr_scheduler.step()


        #* Save model performing best on test dataset
        if best_accuracy < epoch_accuracy:
            best_accuracy = epoch_accuracy
            print("New best test accuracy! : " + str(round(best_accuracy, 3)))

        #     torch.save(deepcaps.state_dict(), checkpoint_path)
        #     print("Saved model at epoch %d"%(epoch_idx))


    accuracy_folder = './results'
    np.save(os.path.join(accuracy_folder, 'training_loss'), train_loss_list, allow_pickle=True)
    np.save(os.path.join(accuracy_folder, 'training_acc'), train_acc_list, allow_pickle=True)
    np.save(os.path.join(accuracy_folder, 'testing_loss'), test_loss_list, allow_pickle=True)
    np.save(os.path.join(accuracy_folder, 'testing_acc'), test_acc_list, allow_pickle=True)
    np.save(os.path.join(accuracy_folder, 'training_r2_2'), train_r2_list, allow_pickle=True)
    np.save(os.path.join(accuracy_folder, 'testing_r2_2'), test_r2_list, allow_pickle=True)


    print('Finished Training')