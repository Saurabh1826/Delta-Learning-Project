import math
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython import embed

# input size (x):  b x ctxt x lat x long   -- ctxt is context length
# hidden size (h): b x cout x lat x long   -- cout should be ==ctxt for CLSTM composition
# cell state (c):  b x cout x lat x long

def batchify_data(data, bsz):
    ## reformat data so that batches work for sequential data
    nbatch = data.shape[0]//bsz
    data = data.narrow(0, 0, nbatch*bsz)
    data = data.view(nbatch, bsz, *data.shape[1:]).contiguous()
    return data

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
        
def viz_eval(model, data, lat, lon, num, filename, figsize=(7,5), diff=False, theory=0):
    warm_up = 2
    h = None
    for i in range(warm_up):
        xpred,h = model(data[i:i+1,:,...], h)        

    samples = np.random.choice(data.shape[0]-1, size=num)
    fig = plt.figure(figsize=figsize)

    for i, idx in enumerate(samples):
        xtest = data[idx+1,0,...]

        if diff: #predicting difference between theory and actual
            xpred,h = model(data[idx:idx+1,:,...],h)

            xtheory = theory[idx+1,0,...]
            xpred = xpred + xtheory #(actual theory) + (pred diff)
            xtest = data[idx+1,0,...] + xtheory  #(actual diff) + (actual theory)
            xtheory = xtheory.cpu().detach().numpy().reshape(xtheory.shape[-2:])
        else:
            xpred,h = model(data[idx:idx+1,...], h)

        xtest = xtest.cpu().detach().numpy().reshape(xtest.shape[-2:])
        xpred = xpred.cpu().detach().numpy().reshape(xpred.shape[-2:])

        cols = 3 if diff else 2
        ax = plt.subplot(num, cols, int(cols*i)+1)
        visualize(xtest, lat, lon, title=None)
        ax.set_title("Actual")
        ax = plt.subplot(num, cols, int(cols*i)+2)
        visualize(xpred, lat, lon, title=None)
        ax.set_title("Predicted")
        if diff:
            ax = plt.subplot(num, cols, int(cols*i)+3)
            visualize(xtheory, lat, lon, title=None)
            ax.set_title("Theory")
    plt.savefig(filename+'.png')

def viz_seq(data, lat, lon, num, filename, figsize=(20, 5)):
    samples = data[:num,...]
    fig = plt.figure(figsize=figsize)
    for i in range(num):
        ax = plt.subplot(1, num, i+1)
        visualize(data[i,...], lat, lon, title=None)
    plt.savefig(filename+'.png')


def spatial_smoothness_loss(output, h_w, w_w):
    """
    Compute the spatial smoothness loss where last 2 dims are image
    Args:
        output (torch.Tensor): Predicted tensor of shape 
                               (batch_size, features, height, width)
    Returns:
        torch.Tensor: Smoothness loss (scalar)
    """
    # Compute spatial gradients
    diff_h = torch.diff(output, dim=-2)  # Difference along height
    diff_w = torch.diff(output, dim=-1)  # Difference along width
    
    # Compute smoothness loss as L2 norm of spatial gradients
    loss_h = torch.norm(diff_h, p=2)
    loss_w = torch.norm(diff_w, p=2)
    
    # Combine losses
    smoothness_loss = h_w*loss_h + w_w*loss_w
    # print(smoothness_loss)
    
    return smoothness_loss

def spatial_loss(pred):
    """Encourage structured outputs"""
    diff_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    diff_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    return diff_x.mean() + diff_y.mean()

def var_loss(pred, target):
    """Penalize the difference in variance for every row and column"""
    row_var_pred = torch.var(pred, dim=-2)
    row_var_target = torch.var(target, dim=-2)
    col_var_pred = torch.var(pred, dim=-1)
    col_var_target = torch.var(target, dim=-1)

    row_loss = torch.mean((row_var_pred - row_var_target)**2)
    col_loss = torch.mean((col_var_pred - col_var_target)**2)

    return row_loss + col_loss
    

class ConvLSTMUnit(nn.Module):
    '''
    Class for one ConvLSTM unit. Can be stacked for more complex networks.
    bias [bool] -- whether or not to add bias
    '''
    def __init__(self, in_dim, h_dim, kernel_size, bias):
        super(ConvLSTMUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_dim+h_dim,
                              out_channels=4*h_dim,
                              kernel_size=kernel_size,
                              padding='same', #(kernel_size[0]//2, kernel_size[1]//2),
                              bias=bias)
        init.constant_(self.conv.bias, 1.0)
        self.h_dim = h_dim
        self.relu = nn.ReLU()

    def forward(self, x, h, c):
            
        xh_cat = torch.cat([x, h], dim=1)
        conv_out = self.conv(xh_cat)

        # split into gate outputs
        # forget, input, cand. memory, output
        fgate, igate, cand_mem, ogate = torch.split(conv_out, self.h_dim, dim=1)
        
        fgate = torch.sigmoid(fgate)
        igate = torch.sigmoid(igate)
        cand_mem = torch.tanh(cand_mem)
        ogate = torch.sigmoid(ogate)

        c_new = fgate*c + igate*cand_mem
        h_new = ogate*torch.tanh(c_new)

        # print("Forget Gate:", fgate.mean().item(), fgate.std().item())
        # print("Input Gate:", igate.mean().item(), igate.std().item())
        # print("Output Gate:", ogate.mean().item(), ogate.std().item())
        # print("Candidate Memory:", cand_mem.mean().item(), cand_mem.std().item())
        
        return h_new, c_new

    def init_hidden(self, batch, im_shape):
        ht, wd = im_shape
        return (torch.randn(batch, self.h_dim, ht, wd, device=self.conv.weight.device),
                torch.randn(batch, self.h_dim, ht, wd, device=self.conv.weight.device))
    
#######################
## Encode-Forecast scheme is a composition of ConvLSTM units
## referencing this paper: https://arxiv.org/pdf/1506.04214v2
    
class ConvLSTM(nn.Module):

    def __init__(self, dim_array, final_out, device, use_forecast=False):
        '''
        dim_array of format:
            [in_dim, h_dim, kernel_size, bias (1/0)] for each twinned ConvLSTM
            Each row is instantiated as a ConvLSTMUnit twice, once for encoding and once for forecasting
        
        context_len: length of prediction "chunk"
        '''
        super(ConvLSTM, self).__init__()
        self.device = device
        self.use_forecast = use_forecast

        self.n_layers = dim_array.shape[0]

        ### Build encoder and forecaster
        self.encode = nn.ModuleList()
        self.forecast = nn.ModuleList()
        for i, row in enumerate(dim_array):
            self.encode.append(ConvLSTMUnit(*map(int, row)))
            bn_size = row[1]
            self.encode.append(nn.BatchNorm2d(bn_size))
            self.encode.append(nn.ReLU())
            row_f = row
            row_f[0] = row[1] #output of encode layer is input of forecast layer
            self.forecast.append(ConvLSTMUnit(*map(int, row_f)))
            self.forecast.append(nn.BatchNorm2d(bn_size))
            self.forecast.append(nn.ReLU())

        self.final = nn.Conv2d(in_channels=dim_array[-1, 1],
                               out_channels=final_out,
                               kernel_size=1,
                               bias=True)

        self.encode = self.encode.to(device)
        self.forecast = self.forecast.to(device)
        self.final = self.final.to(device)
        self.n_units = 3
        self.relu = nn.ReLU()

    def forward(self, x, hcs=None):

        if hcs==None: # all the hidden and cell states of all LSTMs
            hcs = self.init_hiddens(x.shape)
        
        for i in range(len(hcs)):
            for j in range(len(hcs[i])):
                hcs[i][j] = hcs[i][j].detach()

        h_es, c_es, h_fs, c_fs = hcs
        x_e = x
        for layer_idx in range(self.n_layers):
            h_e = h_es[layer_idx].detach()
            h_f = h_fs[layer_idx].detach()
            c_e = c_es[layer_idx].detach()
            c_f = c_fs[layer_idx].detach()
            
            # Encoding step
            h_e, c_e = self.encode[self.n_units*layer_idx](x_e, h_e, c_e)

            # Forecasting step
            x_f, c_f = h_e, c_e
            if self.use_forecast:
                h_f, c_f = self.forecast[layer_idx](x_f, h_f, c_f)

            x_e = h_e
            h_es[layer_idx] = h_e
            h_fs[layer_idx] = h_f
            c_es[layer_idx] = c_e
            c_fs[layer_idx] = c_f

            h_final = h_e if not self.use_forecast else h_f

        return self.final(h_final), [h_es, c_es, h_fs, c_fs]
        
    def init_hiddens(self, x_shape):
        im_shape = x_shape[-2:]
        batch = x_shape[0]
        h_es, c_es, h_fs, c_fs = [], [], [], []

        for i in range(self.n_layers):
            h_e, c_e = self.encode[self.n_units*i].init_hidden(batch, im_shape)
            h_f, c_f = self.forecast[self.n_units*i].init_hidden(batch, im_shape)
            h_es.append(h_e)
            c_es.append(c_e)
            h_fs.append(h_f)
            c_fs.append(c_f)
        
        return [h_es, c_es, h_fs, c_es]

def train(model, optimizer, train_data, bsz, epochs=10):
    model.train()
    loss_fn = nn.MSELoss()
    batches = train_data.shape[0] - 1
    hidden = None

    scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=10,min_lr=1e-6)
    
    for epoch in range(epochs):

        for idx in range(batches):
            xbatch = train_data[idx]
            
            optimizer.zero_grad()
            xpred, hidden = model(xbatch, hidden)

            smoothing_factors = (0.0, 0.0) # penalizes smoothness along height, along width
            
            xnext = train_data[idx+1, :, 0,...].unsqueeze(dim=1) #get_batch(train_data, idx+1, bsz)
            mseloss = loss_fn(xpred, xnext)
            # regloss = spatial_smoothness_loss(xpred, *smoothing_factors)
            spatloss = spatial_loss(xpred)
            # uniformity_penalty = torch.abs(torch.var(xnext) - torch.var(xpred))
            uniformity_penalty = 1/(torch.var(xpred)+1e-6) #var_loss(xpred, xnext) #1/(torch.var(xpred)+0.01)
            mean_penalty = torch.abs(torch.mean(xpred) - torch.mean(xnext))
            
            loss = mseloss + 0.1*spatloss + 3*uniformity_penalty + 0.1*mean_penalty
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # clip norm
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.) #0.25)
            optimizer.step()

            scheduler.step(loss)

        print('EPOCH:', epoch, 'LOSS:', loss.item(), 'LR:', scheduler.get_last_lr())
        print('VAR:', uniformity_penalty.item(), 'MEAN:', mean_penalty.item(), 'SPAT:', spatloss.item(), 'MSE:', mseloss.item())
        # print('total grad norm:', total_norm)

    return model
    
if __name__=='__main__':
    theory_model = 'base'
    predict_diff = True
    use_airtemp = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xrds=xr.open_dataset("combined_data.nc")
    df1=xrds.to_dataframe()

    stl_data = xrds['stl1'].to_numpy()
    t2m_data = xrds['t2m'].to_numpy()
    latitude = xrds['latitude'].to_numpy()
    longitude = xrds['longitude'].to_numpy()

    stl_tensor = torch.Tensor(stl_data)
    t2m_tensor = torch.Tensor(t2m_data)
    stl_tensor = torch.unsqueeze(stl_tensor, dim=1)
    t2m_tensor = torch.unsqueeze(t2m_tensor, dim=1)
    
    ### Load pre-calculated theoretical temperatures
    base_model = torch.Tensor(np.load('base_model.npy'))
    vegetation_model = torch.Tensor(np.load('vegetation_model.npy'))

    ## all nans are columns 51 onward
    stl_tensor = stl_tensor[:,:,:,:51]
    t2m_tensor = t2m_tensor[:,:,:,:51]
    longitude = longitude[...,:51]
    
    ### Concatenate theory model to input
    if theory_model == 'base':
        m = base_model
    else:
        m = vegetation_model
    m = m.permute(2,0,1)
    m = m[...,:51]
    # m = impute_nans_with_local_mean(m)
    theory_x = m.unsqueeze(dim=1) # exists from T1 onward

    if predict_diff:
        soil_data = stl_tensor[1:,...] - theory_x
        x = soil_data
        s = soil_data.squeeze(1).cpu().numpy(); viz_seq(s, latitude, longitude, 4, 'diff_seq')
        if use_airtemp:
            x = torch.cat((soil_data[:-1,...], t2m_tensor[2:,...]), dim=1)# theory_x[1:,...]), dim=1) # each row is (current soil diff, future air temp, future theory)
    else:
        x = stl_tensor
        if use_airtemp:
            x = torch.cat((stl_tensor[:-1,...], t2m_tensor[1:,...]), dim=1)
    x = x.to(device)
    
    # Params of (in_channels, out_channels, kernel_size, bias(bool)) per layer
    dim_array = np.array([[x.shape[1], 128, 3, 1]]).astype(int)
                          # [128, 256, 3, 1]]).astype(int)
    model  = ConvLSTM(dim_array, final_out=1, device=device)
    model = model.to(device)

    temp_normalization = 253
    if predict_diff:
        x[:,1:,...] -= temp_normalization #only normalize air temp and theory soil, not diff
    else:
        x -= temp_normalization

    bsz = 64
    x = batchify_data(x, bsz)
    
    ### Train model
    train_split = int(0.9 * x.shape[0])
    train_data = x[:train_split, :]
    val_data = x[train_split:, :]
    lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    # ckpt = torch.load('clstmdiff.pth')
    # model.load_state_dict(ckpt['state_dict'])
    model = train(model, optimizer, train_data, bsz=bsz, epochs=150)
    

    ### Visualize results
    try:
        if predict_diff:
            train_data = train_data.reshape(-1,*train_data.shape[2:])
            viz_eval(model, train_data, latitude, longitude, 4, 'clstm_diff', figsize=(20,20), diff=False)
            viz_eval(model, train_data, latitude, longitude, 4, 'clstm_diff_theory', figsize=(20,20), diff=True, theory=theory_x[1:,...].to(device))
            
        else:
            train_data = train_data.reshape(-1,*train_data.shape[2:])
            viz_eval(model, train_data, latitude, longitude, 4, 'clstm_nodiff', figsize=(20,20), diff=predict_diff)

    except Exception as e:
        print('viz error')
        print(e)
        embed()
    embed()
