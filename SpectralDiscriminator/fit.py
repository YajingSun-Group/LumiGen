import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from scipy.stats import pearsonr   
import random
import os
import time
import pandas as pd

def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def coo_format(A):
    coo_A = np.zeros([A.shape[0],A.shape[2]])
    for i in range(A.shape[1]):
        coo_A = coo_A + A[:,i,:]
    coo_A = coo_matrix(coo_A)
    edge_index = [coo_A.row, coo_A.col]
    edge_attr = []
    for j in range(len(edge_index[0])):
        edge_attr.append(A[edge_index[0][j],:,edge_index[1][j]])
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr

def func(pred):
    return pred.index(max(pred))

def train(model, train_loader, device, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, att = model(data)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        
        optimizer.step()
    return loss_all/len(train_loader.dataset)

def test(model, loader, device, mean, std):
    model.eval()
    error = 0
    loss_all = 0
    model_output = []
    y = []
    att_list = []
    for data in loader:
        data = data.to(device)
        output, att = model(data)
        error += (output * std - data.y * std).abs().sum().item()           #mae
                                                                                   
        loss = F.mse_loss(torch.tensor([i*std+mean for i in output]), torch.tensor([i*std+mean for i in data.y]))   
        loss_all += loss.item() * data.num_graphs       #   Returns a new Tensor, detached from the current graph. The result will never require gradient.
        model_output.extend(output.tolist())                                 
        y.extend(data.y.tolist())
        att_list.extend(att.tolist())
    return loss_all/len(loader.dataset), error/len(loader.dataset), model_output, y, att_list


def training(Model, data_loaders, n_epoch=400, snapshot_path='./snapshot/', save_att=False, optimizer=None, std=None, mean=None,args=None):
    '''
    data_loaders is a dict that contains Data_Loaders for training, validation and testing, respectively
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.00001)
    
    if len(data_loaders)>2:
        with_test = True
    else:
        with_test = False

    if args.mode == 'fine_tuning':
        print('fine_tuning')
        model_path = 'snapshot_Bayes_Opt/save_path/{}_ModelParams.pkl'.format(args.test_csv)
        model.load_state_dict(torch.load(model_path))       
        
    history = {}
    history['Train Loss'] = []     
    history['Train Mae'] = []
    history['Train Rpcc'] = []
    history['Train Rpcc2'] = []
    history['Train R2'] = []
    
    history['Valid Loss'] = []
    history['Valid Mae'] = []
    history['Valid Rpcc'] = []
    history['Valid Rpcc2'] = []
    history['Valid R2'] = []
    

    reports = {}
    reports['valid mae'] = 0.0
    reports['valid loss'] = float('inf')
    reports['valid R2'] = 0.0
    for epoch in range(1, n_epoch+1):
        start_time_1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(model, data_loaders['train'], device, optimizer) 
        train_loss, train_mae, train_output, y_train, _ = test(model, data_loaders['train'], device, mean, std)
        valid_loss, valid_mae, valid_output, y_valid, valid_att = test(model, data_loaders['valid'], device, mean, std)
        
        train_pccs = pearsonr(y_train, train_output)[0]
        train_r2_pccs = train_pccs**2
        train_r2 = r2_score(y_train, train_output)
        
        valid_pccs = pearsonr(y_valid, valid_output)[0]
        valid_r2_pccs = valid_pccs**2
        valid_r2 = r2_score(y_valid, valid_output)
        
        scheduler.step(epoch)
        history['Train Loss'].append(train_loss)
        history['Train Mae'].append(train_mae)
        history['Train Rpcc'].append(train_pccs)
        history['Train Rpcc2'].append(train_r2_pccs)
        history['Train R2'].append(train_r2)
        
        history['Valid Loss'].append(valid_loss)
        history['Valid Mae'].append(valid_mae)
        history['Valid Rpcc'].append(valid_pccs)
        history['Valid Rpcc2'].append(valid_r2_pccs)
        history['Valid R2'].append(valid_r2)
        
        if valid_loss < reports['valid loss']:
            verify_dir_exists(snapshot_path)
            torch.save(model.state_dict(), snapshot_path+'/{}_ModelParams.pkl'.format(args.test_csv))
            if save_att:
                if with_test:
                    test_mae, test_loss, test_output, y_test, test_att = test(model, data_loaders['test'], device, mean, std)
                    test_r2 = r2_score(y_test, test_output)
                    test_MAE = mean_absolute_error(y_test, test_output)
                    test_MSE = mean_squared_error(y_test, test_output)
                    test_RMSE = test_MSE ** 0.5
                    #open(snapshot_path+'/model-test-info.txt'.format(epoch), 'w').write('\n'.join(['test R2:{}'.format(str(test_r2)), 'test mae:{}'.format(str(test_mae))]))
                    data_csv = {'y_test_csv': [i*std+mean for i in y_test], 'test_output_csv': [i*std+mean for i in test_output]}
                    df = pd.DataFrame(data_csv)
                    df.to_csv('{}_{}_test_result.csv'.format(args.train_csv, args.test_csv), index=None)
                    from metric import compute_metrics
                    metrics_mae, metrics_mse, metrics_r2, metrics_rmse = compute_metrics('{}_{}_test_result.csv'.format(args.train_csv, args.test_csv))
                    #print('test R2:',metrics_r2,'test mae:',metrics_mae,'test_MSE:',metrics_mse,'test_RMSE:',metrics_rmse)

            else:
                open(snapshot_path+'/model-{}_info.txt'.format(epoch), 'w').write('\n'.join(['Step:{}'.format(epoch), 'train mae:{}'.format(str(train_mae)), \
                                                                                             'train loss:{}'.format(str(train_loss)), 'train r2_pccs:{}'.format(str(train_r2_pccs)), \
                                                                                             'train R2:{}'.format(str(train_r2)), 'valid mae:{}'.format(str(valid_mae)), \
                                                                                             'valid loss:{}'.format(str(valid_loss)), 'valid r2_pccs:{}'.format(str(valid_r2_pccs)), \
                                                                                             'valid R2:{}'.format(str(valid_r2)), 'y_train:{}'.format(str(y_train)), \
                                                                                             'y_train_pred:{}'.format(str(train_output)), 'y_valid:{}'.format(str(y_valid)), \
                                                                                             'y_valid_pred:{}'.format(str(valid_output))]))
                data_csv_train = {'y_train_csv': [i*std+mean for i in y_train], 'train_output_csv': [i*std+mean for i in train_output]}
                df_t = pd.DataFrame(data_csv_train)
                df_t.to_csv('{}_{}_train_result.csv'.format(args.train_csv, args.test_csv), index=None)
                data_csv_valid = {'y_valid_csv': [i*std+mean for i in y_valid], 'valid_output_csv': [i*std+mean for i in valid_output]}
                df_v = pd.DataFrame(data_csv_valid)
                df_v.to_csv('{}_{}_valid_result.csv'.format(args.train_csv, args.test_csv), index=None)
                if with_test:
                    test_mae, test_loss, test_output, y_test, test_valid = test(model, data_loaders['test'], device, mean, std)
                    test_r2 = r2_score(y_test, test_output)
                    #open(snapshot_path+'/model-test-info.txt'.format(epoch), 'w').write('\n'.join(['test R2:{}'.format(str(test_r2)), 'test mae:{}'.format(str(test_mae))]))
                    # print('test R2:',test_r2,'test mae:',test_mae)
                    df = pd.DataFrame(([[i*std+mean for i in y_test],[i*std+mean for i in test_output]]),index=False)
                    df.to_csv('{}_{}_test_result.csv'.format(args.train_csv, args.test_csv), index=None)
                    from metric import compute_metrics
                    metrics_mae, metrics_mse, metrics_r2, metrics_rmse = compute_metrics('{}_test_result.csv'.format(args.test_csv))
                    #print('test R2:',metrics_r2,'test mae:',metrics_mae,'test_MSE:',metrics_mse,'test_RMSE:',metrics_rmse)

            reports['valid mae'] = valid_mae
            reports['valid loss'] = valid_loss
            reports['valid R2'] = valid_r2        
        end_time_1 = time.time()
        elapsed_time = end_time_1 - start_time_1
        print('Epoch:{:03d} ==> LR: {:7f}, Train loss: {:.4f}, Train R2_pccs: {:.4f}, Train_R2: {:.4f}, Valid loss: {:.4f}, Valid R2_pccs: {:.4f}, Valid R2: {:.4f}, Elapsed Time:{:.2f} s'.format(epoch, 
                                           lr, 
                                           train_mae, 
                                           train_r2_pccs,
                                           train_r2, 
                                           valid_mae, 
                                           valid_r2_pccs, 
                                           valid_r2,
                                           elapsed_time))
    open(snapshot_path+'history','w').write(str(history))
    print('\nLoss_mae: {}'.format(reports['valid mae'] ))
    return reports
