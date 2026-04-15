

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import copy
import numpy as np

from model import *



class likelihood(nn.Module):
    def __init__(self):
        super(likelihood, self).__init__()
    def forward(self, label, mu, sigma):
        distribution = torch.distributions.normal.Normal(mu,sigma)
        loss = distribution.log_prob(label)
        return -torch.mean(loss)



def train_parameter(args, model, train_loader, val_loader,dir_best_model='../Model/best_ann.pt'):
    best_val_loss = float('inf')
    counter = 0
    lr=args.lr
    device=args.device
    num_epochs=args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = likelihood()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            inputs=inputs.reshape(-1,inputs.shape[-1])
            labels=labels.reshape(-1,1)
            mu, sigma = model(inputs)
            mu = mu.reshape(-1,1)   
            sigma = sigma.reshape(-1,1)
            loss = criterion(labels,mu,sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss}')
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                inputs=inputs.reshape(-1,inputs.shape[-1])
                labels=labels.reshape(-1,1)
                mu, sigma = model(inputs)
                mu = mu.reshape(-1,1)   
                sigma = sigma.reshape(-1,1)
                loss = criterion(labels,mu,sigma)
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dir_best_model)
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(best_val_loss)
                break
    model.load_state_dict(torch.load(dir_best_model))



def train(args, model, train_loader, val_loader,dir_best_model='../model/best_ann.pt'):
    best_val_loss = float('inf')
    counter = 0
    lr=args.lr
    device= args.device
    print(f"Using device: {device}")
    num_epochs=args.num_epochs
    patience=args.patience
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)  # 添加weight_decay参数
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dir_best_model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(best_val_loss)
                break

    model.load_state_dict(torch.load(dir_best_model))




def DFL_train(args, combined_model, combined_fine_tune_loader, combined_test_loader, scaler_y_load, time_flag=False):
    device = args.device
    
    optimizer = torch.optim.SGD([
        {'params': combined_model.model_load.parameters()}
    ], lr=args.e2e_lr)

    for i in range(3):
        print('===============================')
        print("e2e epoch:", i)
        print('===============================')
        for input_load, labels_load, alpha, alpha_z, rho_z, rho_r in combined_fine_tune_loader:
            combined_model.train()
            input_load = input_load.to(device).float()
            labels_load = labels_load.to(device).float()
            alpha = alpha.to(device).float()
            alpha_z = alpha_z.to(device).float()
            rho_r = rho_r.to(device).float()    
            rho_z = rho_z.to(device).float()
            # ==== 嵌入 E2E_fine_tuning_parameter 的内容 ====
            optimizer.zero_grad()
            # reshape
            input_load = input_load.reshape(-1, input_load.shape[-1])
            # forward预测
            pred_load = combined_model.model_load_forward(input_load, scaler_y_load)
            # 数据归一化反变换
            if isinstance(scaler_y_load, StandardScaler):
                mean = torch.tensor(scaler_y_load.mean_, dtype=torch.float64).to(device)
                scale = torch.tensor(scaler_y_load.scale_,  dtype=torch.float64).to(device)
                labels_load = labels_load * scale + mean
            # 判断 MinMaxScaler
            elif isinstance(scaler_y_load, MinMaxScaler):
                data_min = torch.tensor(scaler_y_load.data_min_,  dtype=torch.float64).to(device)
                data_max = torch.tensor(scaler_y_load.data_max_, dtype=torch.float64).to(device)
                labels_load = labels_load * (data_max - data_min) + data_min
            # forward优化目标
            solution_ahead, solution_intra, obj = combined_model.forward(args, pred_load, labels_load, alpha, alpha_z, rho_z, rho_r)
            # loss & backward & optimizer
            loss = obj.mean()
            loss.backward()
            optimizer.step()

    return combined_model

