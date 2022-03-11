#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:38:23 2021

@author: liujiachen
"""
import os, time
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
    
def file_scanner_loss(file, mode = None, x_axis = 'epoch'):

    loss_list = defaultdict(list)
    # epoch_list = []
    participants_list = defaultdict(int)
    if not os.path.isdir(file):
        f = open(file) 
        
        iter_f = iter(f) 
        for line in iter_f: 
            if line.find('Training loss') > -1: 
                loss=   float( re.compile('Training loss: ([0-9]*.[0-9]*)').findall(line)[0])
                epoch = int( re.compile('Epoch: ([0-9]*)').findall(line)[0])
                participants = int( re.compile('Planned participants: ([0-9]*)').findall(line)[0])
                loss_list[epoch].append(loss*participants)
                participants_list[epoch] += participants
    avg_loss_epoch =  [np.sum(loss_list[k] )/ max( participants_list[k], 1) for k in loss_list ]
    return list(loss_list.keys()), avg_loss_epoch



def file_scanner_acc(file, mode = 'top1', x_axis = 'epoch'):
    acc_list = []
    epoch_list = []
    separate_acc = defaultdict(list) 
    if not os.path.isdir(file):
        f = open(file) 
        
        iter_f = iter(f) 
        cnt = 0
        for line in iter_f: 
            if line.find("top_1") > -1: 
                if mode=='top1':
                    acc = float( re.compile('top_1: ([0-9]*.[0-9]*)').findall(line)[0])
                else:
                    acc = float( re.compile('top_5: ([0-9]*.[0-9]*)').findall(line)[0])
                epoch = int( re.compile('epoch: ([0-9]*)').findall(line)[0])
                print(f"Epoch {epoch} {mode} accuracy：{acc}")
                acc_list.append(acc)
                epoch_list.append(epoch)
            elif line.find('FL Testing in epoch:') > -1: 
                acc = float( re.compile('top_1: ([0-9]*.[0-9]*)').findall(line)[0])
                epoch = int( re.compile('epoch: ([0-9]*)').findall(line)[0])
                separate_acc[epoch].append(acc)
                
        low = []
        high = []
        if len( separate_acc[epoch]) > 1:
            for e in separate_acc:
                low.append(min( separate_acc[e] ) )
                high.append( max(separate_acc[e]) )
        
        if x_axis == 'wall':
            epoch_id = 0
            f2 = open(file) 
            iter_f2 = iter(f2) 
            wall_clock_list = defaultdict(list) 
            client_list = defaultdict(int) 
            
            for line in iter_f2: 
                if line.find("Wall clock: ") > -1 and line.find("Epoch: "+str(epoch_list[epoch_id] ))   > -1: 
                    # cluster_id = int(re.compile('ClusterID: ([0-9]*)').findall(line)[0])
                    wall_clock = int(re.compile('Wall clock: ([0-9]*)').findall(line)[0])
                    client_num = int(re.compile('Planned participants: ([0-9]*)').findall(line)[0])
                    client_list[ epoch_list[epoch_id] ] += client_num
                    wall_clock_list[ epoch_list[epoch_id] ] .append( wall_clock * client_num)
                    
                    epoch_id += 1
                    if epoch_id >= len(epoch_list):
                        break
            wall_ist =[]
            for epoch in wall_clock_list:
                wall_ist.append( np.sum(wall_clock_list[epoch] ) /client_list[epoch ]  )
                
            epoch_list = wall_ist
    print(epoch_list, acc_list)
    return epoch_list, acc_list, low, high

def file_scanner_proportionality(file, mode =  None, x_axis = None):
    prop_list =  defaultdict(list)
    if not os.path.isdir(file):
        f = open(file) 
        iter_f = iter(f) 
        for line in iter_f: #遍历文件，一行行遍历，读取文本
            if line.find("[Epoch ") > -1: 
               epoch = int(re.compile('\[Epoch ([0-9]*)').findall(line)[0])
               train_time = int(re.compile('train_times ([0-9]*)').findall(line)[0])
               acc = float( re.compile('test_accuracy ([0-9]*.[0-9]*)').findall(line)[0])
               prop_list [epoch].append( acc / (train_time +1))
               
        prop_epoch_list = []
        for epoch in prop_list:
            var = np.var( prop_list [epoch] )
            prop_epoch_list.append(var)
        print(f"File {file} variance of acc/res is {var}")
    
    return list(prop_list.keys()), prop_epoch_list
    
def file_scanner_fairness(file, mode = None, x_axis = None):
    #[ [[], []],
    #  [[], []] ]
    acc_list =  defaultdict(list)
    
    if not os.path.isdir(file):
        f = open(file) 
        iter_f = iter(f) 
        for line in iter_f: #遍历文件，一行行遍历，读取文本
            if line.find("[Epoch ") > -1: 
               epoch = int(re.compile('\[Epoch ([0-9]*)').findall(line)[0])
               # cluster_id  = int(re.compile(', Cluster_ID ([0-9]*)').findall(line)[0])
               
               acc = float( re.compile('test_accuracy ([0-9]*.[0-9]*)').findall(line)[0])
               #train_time = int(re.compile('train_times ([0-9]*)').findall(line)[0])
               acc_list[epoch].append(acc * 100) 
               
            
    var_list = []
    for epoch in acc_list:
        var = np.var( acc_list[epoch])
        worst = np.percentile(acc_list[epoch], 10)
        top = np.percentile(acc_list[epoch], 90)
        var_list.append(var)
        print(f"Epoch {epoch} client variance: {var}")
        print(f"Epoch {epoch} top 10% : {top}. worst 10%: {worst}")
        
    return list(acc_list.keys()), var_list

def scatter_plot(x, y):
    plt.scatter(x, y, c='b')
    dict_x =defaultdict(list)
    for i, n in enumerate( x ):
        dict_x[n].append(y[i])
    plt.scatter(list(dict_x), [ np.mean(v) for v in list(dict_x.values())] , c='r')
    plt.show()

def running_avg( y_list , avg_interval=3):
    extend_y_list = y_list
    for i in range (avg_interval-1):
        extend_y_list = np.insert(extend_y_list, 0, y_list[0])
    
    running_avg = []
    for i in range (len(y_list)):
        running_avg.append(np.mean(extend_y_list[i:i+avg_interval]  ))
    return running_avg

    
def plot(file_list, ylabel = 'Accuracy', mode = None, x_axis = None, name ='_', avg_interval=3, xlim = None):
    if ylabel == 'Accuracy' :
        get_data_func = file_scanner_acc  
    elif ylabel == 'Loss':
        get_data_func = file_scanner_loss
    elif ylabel == 'Variance':
        get_data_func = file_scanner_fairness
    elif ylabel == 'Proportionality':
        get_data_func = file_scanner_proportionality
    
    plt.figure(figsize=( 6,6))
    xmax = 0
    for j, file in enumerate(file_list):
        print(f"Read file {file}")
        
        file_name = file +".txt"
        file1 = open(file_name,"w") 
        if ylabel == 'Accuracy' :
            epoch_list, y_list, low, high = get_data_func(file, mode, x_axis) 
            file1.write(str(low))
            file1.write(str(high))
        else:
            epoch_list, y_list = get_data_func(file, mode, x_axis)

        file1.write(str(epoch_list))
        file1.write(str(y_list))
        
        file1.close()
        print("Write to ",file_name)

        #y_list = running_avg(y_list, avg_interval)
        plt.plot( epoch_list, y_list, color= "C" + str( j ), label= file)
        
        xmax = max(xmax, max(epoch_list ))
    xmax = xmax if xlim is None else xlim
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim([0, xmax])
    plt.ylabel(ylabel+' '+mode)
    plt.xlabel(x_axis)
    plt.show()
#    plt.title('Averaged model accuracy on global test dataset') 
    fig_name =  str(int(time.time()))+'_'+name+'_acc_result.png'
    plt.savefig(fig_name)
    print("Save to ",fig_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate FL for noniid clients ')
  
    parser.add_argument('--y_axis', type=str, default='acc', 
                        help='plot mode: acc , train-times, prop')
    parser.add_argument('--x_axis', type=str, default='epoch', 
                        help='plot mode: epoch , wall')
    parser.add_argument('--acc_mode', type=str, default='top1', 
                        help='plot mode: acc , train-times')
    parser.add_argument('--path', type=str, default='.', 
                        help='working directory')
    parser.add_argument('--name', type=str, default='_', 
                        help='figure name')
    
    parser.add_argument('--interval', type=int, default=3, 
                        help='running sum')
    
    parser.add_argument('--xlim', type=int, default=None, 
                        help='max epoch')
    parser.add_argument('--file', nargs='+' ,help='log gile name')
    
    args = parser.parse_args()
    os.chdir(args.path)
    if args.y_axis == 'acc':
        ylabel = 'Accuracy'
    elif args.y_axis=='loss':
        ylabel = 'Loss'
    elif args.y_axis=='var':
        ylabel ='Variance'
    elif args.y_axis == 'prop':
        ylabel ='Proportionality'
  
    plot(args.file, ylabel, mode = args.acc_mode, x_axis = args.x_axis , name= args.name,avg_interval=args.interval, xlim = args.xlim)
    
    
