import sys
import rdkit
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import NumValenceElectrons
import rdkit.Chem.Descriptors as dsc
from model import *
import fit

from Mol2Graph import Mol2Graph
from rdkit.Chem import AllChem as Chem
from multiprocessing import Pool
import numpy as np

import torch
import random
from torch_geometric.data import Data #, DataLoader
from torch_geometric.loader import DataLoader
from multiprocessing import Pool
import argparse

import os
    
def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="regression")
    parser.add_argument('--mode', type=str, default='training')
    #"fine_tuning")
    #parser.add_argument('--dir', type=str, help='the table of coformer pairs of train set')
    parser.add_argument('--train_csv', type=str, help='training/validation set')
    parser.add_argument('--test_csv', type=str, help='test set')
    args = parser.parse_args()

    return args
    
class DataFeat(object):
    def __init__(self, **kwargs):
        for k in kwargs:
            self.__dict__[k] = kwargs[k]

def Calc_AROM(mh):
    m = Chem.RemoveHs(mh)
    ring_info = m.GetRingInfo()
    atoms_in_rings = ring_info.AtomRings()
    num_aromatic_ring = 0
    for ring in atoms_in_rings:
        aromatic_atom_in_ring = 0
        for atom_id in ring:
            atom = m.GetAtomWithIdx(atom_id)
            if atom.GetIsAromatic():
                aromatic_atom_in_ring += 1
        if aromatic_atom_in_ring == len(ring):
            num_aromatic_ring += 1
    return num_aromatic_ring
    
def make_data(smiles_list):  
    try:##for i in smiles_list:
        iMol1 = Chem.MolFromSmiles(smiles_list.split(',')[0])
        iMol2 = Chem.AddHs(Chem.MolFromSmiles(smiles_list.split(',')[1]))
        #print('y:',np.array([float(smiles_list.split(',')[2])]))
        g1 = Mol2Graph(iMol1)
        g2 = Mol2Graph(iMol2)
        x = np.concatenate([g1.x, g2.x], axis=0)
        edge_feats = np.concatenate([g1.edge_feats, g2.edge_feats], axis=0)
        e_idx2 = g2.edge_idx+g1.node_num
        edge_index = np.concatenate([g1.edge_idx, e_idx2], axis=0)
        return DataFeat(x=x, edge_feats=edge_feats, edge_index=edge_index, y=np.array([float(smiles_list.split(',')[2])]))
    except: 
        print('Bad input sample:'+smiles_list.split(',')[0]+'skipped.')
        
if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_argv()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 1:
        smiles_data_train = open(args.train_csv)
        smiles_data_test = open(args.test_csv)
        smiles_line = smiles_data_train.readlines()
        random.shuffle(smiles_line)
        split_idx = int(0.9 * len(smiles_line))
        smiles_train = smiles_line[:split_idx]
        smiles_valid = smiles_line[split_idx:]
        smiles_test = smiles_data_test.readlines()
        print('smiles_train:',len(smiles_train),'smiles_test',len(smiles_test))
        pool = Pool(processes=32)                 
        data_pool_Train = pool.map(make_data, [i for i in smiles_train if Chem.MolFromSmiles(i.split(',')[0]) != None and Chem.AddHs(Chem.MolFromSmiles(i.split(',')[1]))!= None]) 
        data_pool_valid = pool.map(make_data, [i for i in smiles_valid if Chem.MolFromSmiles(i.split(',')[0]) != None and Chem.AddHs(Chem.MolFromSmiles(i.split(',')[1]))!= None])   
        data_pool_test = pool.map(make_data, [i for i in smiles_test if Chem.MolFromSmiles(i.split(',')[0]) != None and Chem.AddHs(Chem.MolFromSmiles(i.split(',')[1]))!= None])
        pool.close()
        pool.join()

            ############## make graph data ##############
        Y_Train = np.array([i.y for i in data_pool_Train])
        std = Y_Train.std()
        mean = Y_Train.mean()
        print('std:',std,',mean:',mean)
    
        loader_Train = []
        for d in data_pool_Train:
            i = Data(x=torch.tensor(d.x),  
                 edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                 edge_attr=torch.tensor(d.edge_feats),
                 y=torch.tensor((d.y-mean)/std, dtype=torch.float32))
            loader_Train.append(i)
        #print('loader_Train:', len(loader_Train))

        loader_valid = []
        for d in data_pool_valid:
            i = Data(x=torch.tensor(d.x),  
                 edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                 edge_attr=torch.tensor(d.edge_feats),
                 y=torch.tensor((d.y-mean)/std, dtype=torch.float32))
            loader_valid.append(i)

        loader_test = []
        for d in data_pool_test:
            i = Data(x=torch.tensor(d.x),  
                 edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                 edge_attr=torch.tensor(d.edge_feats),
                 y=torch.tensor((d.y-mean)/std, dtype=torch.float32))
            loader_test.append(i)

    ############## loader data ##############
        random.seed(1000)
        random.shuffle(loader_Train)
        #print('loader_Train:', len(loader_Train))
        if args.mode == 'training':
            BS = 128
            print('mode:training')
        else:
            BS = 32
            print('mode:fine_tuning')
        train_loader = DataLoader(loader_Train, batch_size=BS, shuffle=0, drop_last=True)
        valid_loader = DataLoader(loader_valid, batch_size=BS, shuffle=0)
        test_loader = DataLoader(loader_test, batch_size=BS, shuffle=0)

        ############## fit ##############
        data_loaders = {'train':train_loader, 'valid':valid_loader, 'test':test_loader}
        if args.mode == 'training':
            fit.training(CCPGraph, data_loaders, n_epoch=300, save_att=True, snapshot_path='./snapshot_Bayes_Opt/{}//'.format('save_path'), mean=mean, std=std,args=args)
        else:
            fit.training(CCPGraph, data_loaders, n_epoch=300, save_att=True, snapshot_path='./snapshot_Bayes_Opt/{}//'.format('fine_tuning'), mean=mean, std=std,args=args)
 
