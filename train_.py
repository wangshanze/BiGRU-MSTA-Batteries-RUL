import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from model_ import BiGRUModel
import time

split_namee = '37'
data_namee = '18'
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader(batch_size, workers=2):
    train_set = load(f'train_set_55_{data_namee}_1')
    train_label = load(f'train_label_55_{data_namee}_1')
    test_set = load(f'test_set_55_{data_namee}_1')
    test_label = load(f'test_label_55_{data_namee}_1')

    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, test_loader

def model_train(epochs, model, optimizer, loss_function, train_loader, test_loader, device):
    model = model.to(device)
    minimum_mse = 1000.
    best_model = model
    train_mse = []
    test_mse = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_mse_loss = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            train_mse_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        train_av_mseloss = np.average(train_mse_loss)
        train_mse.append(train_av_mseloss)
        print(f'Epoch: {epoch+1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')
        
        with torch.no_grad():
            model.eval()
            test_mse_loss = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                test_loss = loss_function(pre, label)
                test_mse_loss.append(test_loss.item())
            
            test_av_mseloss = np.average(test_mse_loss)
            test_mse.append(test_av_mseloss)
            print(f'Epoch: {epoch+1:2} test_MSE_Loss:{test_av_mseloss:10.8f}')
            
            if test_av_mseloss < minimum_mse:
                minimum_mse = test_av_mseloss
                best_model = model
    
    torch.save(best_model, f'best_model_bigru_{data_namee}_{split_namee}.pt')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    
    plt.plot(range(epochs), train_mse, color = '#FF5733',label = 'train_MSE_loss')
    plt.plot(range(epochs), test_mse, color = '#33A8FF',label = 'test_MSE_loss')
    plt.title('Train and Test MSE_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE_loss')
    plt.legend()
    plt.savefig('train_test_mse_loss.png',dpi=200)
    plt.show()
    print(f'min_MSE: {minimum_mse}')

if __name__ == "__main__":
    train_loader, test_loader = dataloader(16)
    input_dim = 1
    hidden_layer_sizes = [16, 32, 64]
    output_dim = 1
    loss_function = nn.MSELoss()
    learn_rate = 0.0003
    model = BiGRUModel(input_dim, hidden_layer_sizes, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    epochs = 300
    model_train(epochs, model, optimizer, loss_function, train_loader, test_loader, device) 