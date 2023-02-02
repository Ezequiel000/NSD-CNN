import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import keras
LABLE_FILE = 'C:/Users/007303173/Documents/nsd_data/ppdata/subj01/behav/responses.tsv'
FMRI_DIR = 'C:/Users/007303173/Documents/nsd_data/GLM-fmri-data/subj01'
FMRI_DIR_STND = 'C:/Users/007303173/Documents/nsd_data/standardized-betas/subj01'
TRIAL_PER_SESS = 750
# !pip install line_profiler
# %load_ext line_profiler

class StandardizeDirectory:

  def __init__(self, loading_dir, dumping_dir, response_file):
    self.ld = loading_dir
    self.dd = dumping_dir
    self.rs = response_file
    self.indx = 0
    self.fmri_files = self.getDirFiles()
    self.file_handlers ={}
    self.open_files()

  def __del__(self):
    for value in self.file_handlers.items():
      value.close()
  # returns a list of all files in the directory to be standardized
  def getDirFiles(self):
     files = [f for f in os.listdir(self.ld) if 
              os.path.isfile(os.path.join(self.ld, f)) and
              f[-5:] == '.hdf5']
     files.sort()                          
     return files               

  def open_files(self): 
    for file_name in self.fmri_files:
      path = os.path.join(FMRI_DIR, file_name)
      self.file_handlers[file_name] = h5py.File(path, 'r')

  def z_score_voxels(self, sesh, mean, std):
    z_scores = (sesh - mean )/ std
    return z_scores


  # this method removes the unwanted scans from the array passed in
  # Input: an np array of the standadized values of the betas 
  # Output: a dictionary containing (betas, labels, and the offset from zero of the first index)
  def store_data(self,sesh,sesh_num):
    mean = sesh.mean(0)
    std = sesh.std(0)
    std = np.where(std == 0, 0.00001, std ) # this ensures there is no division by zero
    labels = self.filter_labels(sesh_num) 
    indx = labels.index
    group_dict = {'betas':[],'labels':[],'strt_indx':[] }
    group_dict['strt_indx'].append(self.offset)
    for x in range(len(indx)): 
      group_dict['betas'].append(self.z_score_voxels(sesh[indx[x]-self.indx], mean, std))
      group_dict['labels'].append(labels.at[indx[x], 'ISCORRECT'])
    self.offset = self.offset + len(labels)
    self.indx = self.indx + TRIAL_PER_SESS
    return group_dict
    # Input: path of the responses.tsv file
  # Output: dictionary of indexes as keys with a corresponding class lable
  # in put the sesh number to filter , pass the session as an argument 
  def filter_labels(self, sesh_num):
      print('filtering labels...')
      res = pd.read_csv(self.rs, sep='\t')
      res = pd.DataFrame(res)
      res = res.loc[res['SESSION'] == sesh_num]
      res = res.loc[res['ISOLD'] == 1]
      res = res.loc[res['ISCORRECT'] > -1]
      res = res.filter(items = ['ISCORRECT'] )
      return res

  def dump(self):
    x=0
    for f in self.fmri_files:
      sesh_num = f[-7:-5]
      print('begining dump')
      sesh = np.array(self.file_handlers[f]['betas'])
      ready_data = self.store_data(sesh, int(sesh_num))
      # Open a file
      dump_dir= f'{FMRI_DIR_STND}/stdz_{f}' 
      path = os.path.join(self.dd, dump_dir)
      new_betas = h5py.File(path, 'w')
      print(f'...creating file {x+1}....')
      g1 = new_betas.create_group('betas')
      g2 = new_betas.create_group('labels')
      g3 = new_betas.create_group('startIndx')
      temp = np.array(ready_data['betas'])
      g1.create_dataset('b', data = temp)
      temp = np.array(ready_data['labels'])
      g2.create_dataset('l', data = temp)
      temp = np.array(ready_data['strt_indx'])
      g3.create_dataset('i', data = temp)
      new_betas.close()     
      print(f'DUMPED file {x+1}')
      x += 1