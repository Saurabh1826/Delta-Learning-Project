import math
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from theoretical_models import SoilTempDataProcessor, calculate_soil_temp_base_model, calculate_soil_temp_under_vegetation
from IPython import embed

class TempPredictionCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        super(TempPredictionCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(512)

        self.output_conv = nn.Conv2d(512, output_channels, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.output_conv(x)

        return x

def batchify_data(data, bsz):
    ## reformat data so that batches work for sequential data
    nbatch = data.shape[0]//bsz
    data = data.narrow(0, 0, nbatch*bsz)
    data = data.view(nbatch, bsz, *data.shape[1:]).contiguous()
    return data

def get_batch(data, i, bsz):
    if (i+1)*bsz > data.shape[0]:
        return data[i*bsz:,...]
    else:
        return data[i*bsz:(i+1)*bsz, ...]

def visualize(temp, lat, lon, title='Surface Temperature Map', figname=None):
    # Create the plot
    # plt.clf()
    # plt.figure(figsize=(7, 5))
    plt.contourf(lon, lat, temp, cmap='viridis')  # Plot the first time step
    plt.colorbar(label='Temperature (K)')  # Adjust based on the variable's unit
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if title is not None:
        plt.title(title)
    if figname is not None:
        plt.savefig(figname)

def impute_nans_with_row_avg(tensor):
    nan_mask = torch.isnan(tensor)

    row_averages = torch.nanmean(tensor, -1, keepdim=True)
    row_averages_expanded = row_averages.expand_as(tensor)

    tensor[nan_mask] = row_averages_expanded[nan_mask]

    return tensor

def remove_columns_with_nans(tensor):
    nan_mask = torch.isnan(tensor)

    columns_with_nans = nan_mask.any(dim=-1)

    tensor_clean = tensor[~columns_with_nans, :]

    return tensor_clean

def train(model, optimizer, train_data, bsz, epochs=10, decay=True):
    model.train()
    loss_fn = nn.MSELoss()
    batches = train_data.shape[0]-1 #math.ceil(train_data.shape[0] / bsz) - 1
    # batches = math.ceil(train_data.shape[0] / bsz) - 1

    for epoch in range(epochs):
        total_loss = 0

        if (epoch >= 0) and decay:
          for param_group in optimizer.param_groups:
              param_group['lr'] = 1e-3
        if (epoch >= 4) and decay:
          for param_group in optimizer.param_groups:
              param_group['lr'] = 1e-4
        if (epoch >= 6) and decay:
          for param_group in optimizer.param_groups:
              param_group['lr'] = 1e-5

        for idx in range(batches):
            xbatch = train_data[idx] #get_batch(train_data, idx, bsz)
            # xbatch = get_batch(train_data, idx, bsz)

            optimizer.zero_grad()
            # print(xbatch.shape)
            xpred = model(xbatch)
            xnext = train_data[idx+1,:,0:1,...] #train_data[idx+1:idx+bsz+1,0:1,...]
            # xnext = train_data[idx+1:idx+bsz+1,0:1,...]
            # print(xpred.shape)
            # print(xnext.shape)
            # print()

            # if (epoch >= 20):
            #     print(torch.var(xpred))

            loss = loss_fn(xpred, xnext)
            if (epoch >= 15):
                loss = loss + 200 * (1 / torch.var(xpred))
            total_loss += loss.item()
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400.)

            optimizer.step()
            # scheduler.step(loss)

        if epoch % 2 == 0:
            print(f'EPOCH {epoch}/{epochs}, LOSS: {total_loss / batches:.6f}')
            print(f'Total gradient norm: {total_norm:.6f}')

    return model

def viz_eval(model, data, lat, lon, num, filename, figsize=(7,5), diff=False):
    samples = np.random.choice(data.shape[0], size=num)
    fig = plt.figure(figsize=figsize)

    # Reset normalization
    data = data.clone()
    if diff:
        data[:,1:,...] += temp_normalization #only normalize air temp and theory soil, not diff
    else:
        data += temp_normalization


    # print(data.shape)

    for i, idx in enumerate(samples):

        xtest = data[idx+1,0,...]
        xpred = model(data[idx:idx+1,...])

        # print(xtest.shape, xpred.shape)

        with torch.no_grad():
            if diff: #predicting difference between theory and actual
                xpred = data[idx,-1,...] + xpred #(actual theory) + (pred diff)
                xtest = data[idx+1,0,...] + data[idx,-1,...] #(actual diff) + (actual theory)
                xtheory = data[idx,-1,...] #(actual theory)

                print(i, f'Theory vs Actual L1 Loss: {nn.L1Loss()(xtheory, xtest)}')
                print(i, f'Theory vs Predicted L1 Loss: {nn.L1Loss()(xtheory, xpred.squeeze(0).squeeze(0))}')
                print(i, f'Predicted vs Actual L1 Loss: {nn.L1Loss()(xtest, xpred.squeeze(0).squeeze(0))}')
                print()

                xtheory = xtheory.cpu().detach().numpy().reshape(xtheory.shape[-2:])

            if (not diff):
                xpred += temp_normalization #undo normalization
                print(i, f'Actual vs Predicted L1 Loss: {nn.L1Loss()(xtest, xpred.squeeze(0).squeeze(0))}')


            xtest = xtest.cpu().detach().numpy().reshape(xtest.shape[-2:])
            xpred = xpred.cpu().detach().numpy().reshape(xpred.shape[-2:])
            # print(xtest.shape, xpred.shape)

        cols = 3 if diff else 2
        ax = plt.subplot(num, cols, int(cols*i)+1)
        visualize(xtest, lat, lon, title=None)
        ax.set_title("Actual")
        ax = plt.subplot(num, cols, int(cols*i)+2)
        visualize(xpred, lat, lon, title=None)
        ax.set_title("Predicted")
        if diff:
            ax = plt.subplot(num, cols, int(cols*i)+3)
            visualize(xpred, lat, lon, title=None)
            ax.set_title("Theory")
    plt.savefig(filename+'.png')

if __name__=="__main__":

    ####################
    ### Training options

    theory_model = 'base'
    predict_diff = False
    remove_nans = False
    ####################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xrds=xr.open_dataset("combined_data.nc")
    df1=xrds.to_dataframe()

    stl_data = xrds['stl1'].to_numpy()
    t2m_data = xrds['t2m'].to_numpy()
    lai_lv_data = xrds['lai_lv'].to_numpy()
    lai_hv_data = xrds['lai_hv'].to_numpy()
    tp_data = xrds['tp'].to_numpy()
    skt_data = xrds['skt'].to_numpy()
    latitude = xrds['latitude']
    longitude = xrds['longitude']

    stl_tensor = torch.Tensor(stl_data)
    t2m_tensor = torch.Tensor(t2m_data)
    lai_lv_tensor = torch.Tensor(lai_lv_data)
    lai_hv_tensor = torch.Tensor(lai_hv_data)
    skt_tensor = torch.Tensor(skt_data)
    tp_tensor = torch.Tensor(tp_data)
    stl_tensor = torch.unsqueeze(stl_tensor, dim=1)
    t2m_tensor = torch.unsqueeze(t2m_tensor, dim=1)
    lai_lv_tensor = torch.unsqueeze(lai_lv_tensor, dim=1)
    lai_hv_tensor = torch.unsqueeze(lai_hv_tensor, dim=1)
    skt_tensor = torch.unsqueeze(skt_tensor, dim=1)
    tp_tensor = torch.unsqueeze(tp_tensor, dim=1)
    x = torch.cat((stl_tensor, t2m_tensor, lai_hv_tensor, lai_lv_tensor, skt_tensor, tp_tensor), dim=1) ## Soil temperature needs to be at index 0
    x = x.to(device)

    ### Load pre-calculated theoretical temperatures
    base_model = torch.Tensor(np.load('base_model.npy')).to(device)
    vegetation_model = torch.Tensor(np.load('vegetation_model.npy')).to(device)

    # Impute NaNs
    x = impute_nans_with_row_avg(x)
    base_model = base_model.permute(2,0,1)
    vegetation_model = vegetation_model.permute(2,0,1)
    base_model = impute_nans_with_row_avg(base_model)
    vegetation_model = impute_nans_with_row_avg(vegetation_model)

    # print(x.shape)

    ### Concatenate theory model to input
    if theory_model == 'base':
        m = base_model
    else:
        m = vegetation_model
    theory_x = m.unsqueeze(dim=1)
    x = x[1:,...]

    # print(x.shape)
    # print(theory_x.shape)

    if predict_diff:
        x[:,0:1,...] = x[:,0:1,...] - theory_x

    x = torch.cat((x, theory_x), dim=1)

    model = TempPredictionCNN(1, 1)
    model = model.to(device)

    ## Shift channels so that extra features at time t+1 are in the same index as soil temp at time t
    x[:-1,1:2,...] = x[1:,1:2,...]
    x[:-1,2:3,...] = x[1:,2:3,...]
    x[:-1,3:4,...] = x[1:,3:4,...]
    x[:-1,4:5,...] = x[1:,4:5,...]
    x[:-1,5:6,...] = x[1:,5:6,...]
    if (predict_diff):
        x[:-1,6:7,...] = x[1:,6:7,...]
    x = x[:-1,...]

    temp_normalization = 253
    if predict_diff:
        x[:,1:,...] -= temp_normalization #only normalize air temp and theory soil, not diff
    else:
        x -= temp_normalization

    # Remove nan columns if desired
    if remove_nans:
        x = x[:,:,:,:51]
        # longitude = longitude[:51]

    # Modify x to only include soil temperatures and whatever other features we want
    x = x[:,:1,...]
    # print(x.shape)

    bsz = 16
    x = batchify_data(x, bsz=bsz)

    ## Train model
    train_split = int(0.8 * x.shape[0])
    train_data = x[:train_split, :]
    val_data = x[train_split:, :]
    lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model = train(model, optimizer, train_data, bsz=bsz, epochs=200)

    ## Visualize results
    try:
        viz_eval(model, train_data.reshape(-1,*train_data.shape[2:]), latitude, longitude, 4, 'xdiff', figsize=(20,20), diff=predict_diff)
    except:
        print('viz error')
        # embed()
    # embed()

