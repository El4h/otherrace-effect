

# imports
import os
import glob
import numpy as np
import copy
import sys
sys.path.append('/home/elaheh_akbari/new/')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace/models')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace/training')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace/training/utils')

from utils import helper
from sklearn import metrics
import tqdm
import torch
import torchvision
import numpy as np
import copy
# import costum moduls
from utils import helper

import scipy
import h5py
import argparse
import json
import jsonlines as jsonl
import time
from zipfile import BadZipFile
from filelock import Timeout
from filelock import SoftFileLock as FileLock
import re
import random

lock_timeout = 5
acquire_timeout = 30
fixed_seed_value = 0

def get_lesion_units_keys(lesion_data, verbose=False):
    keys = np.sort(list(lesion_data.keys()))
    split_keys = []
    for key in keys:
        if key.startswith('selected_losses') or key.startswith('selected_units'):
            split_keys.append(key.split('/'))
    split_keys = np.array(split_keys)
    ind = np.lexsort(split_keys.T)
    ind = np.lexsort((split_keys[:,2],split_keys[:,0],split_keys[:,1]))

    prev_key = None
    task_count=0
    lesion_units_keys = []
    lesion_losses_keys = []
    for i, k in enumerate(ind):
        key = os.path.join(*split_keys[k])
        if key.startswith('selected_losses'):
            lesion_losses_keys.append(key)
        elif key.startswith('selected_units'):
            lesion_units_keys.append(key)
        if key.startswith('selected_losses') and i > 0 and prev_key.startswith('selected_units') or prev_key==None:
            if verbose==True:
                print()
                print('TASK', task_count)
                print('----------------')
            task_count+=1
        if key.startswith('selected_units'):
            if verbose==True:
                print('----', key)
        else:
            if verbose==True:
                print(key)
        prev_key = key
    return lesion_units_keys, lesion_losses_keys

def get_duplicate_ids(selected_units):
    selected_units_counts = {}
    duplicate_ids = []
    for i, unit in enumerate(selected_units):
        if unit not in selected_units_counts.keys():
            selected_units_counts[unit] = 1
        else:
            selected_units_counts[unit]+=1
            duplicate_ids.append(i)
    return duplicate_ids

def make_unique(units, losses):
    assert len(units) == len(losses), 'units and losses must have equal length'
    
    units = np.array(units)
    losses = np.array(losses)
    
    duplicate_ids = get_duplicate_ids(units)
    if len(duplicate_ids) > 0:
        print('\nMaking Units Unique:', flush=True)
        print('--removing duplicate units:', units[duplicate_ids], flush=True)
        units = np.delete(units, duplicate_ids)
        losses = np.delete(losses, duplicate_ids)
    else:
        pass
        #print('Values already_unique, no operation necessary.', flush=True)
    print(flush=True)
        
    return units, losses

def conclude_lesion_to_json(filename, sort_task, param_group_index):
    key = os.path.join('status','SORTEDBY_' + sort_task, str(param_group_index))
    lesion_data = get_lesion_data(filename)
    if key not in lesion_data.keys():
        count=1
        write_to_json(filename=filename, writer_method='a', keys=[key], values=['complete'])
    elif lesion_data[key]=='complete':
        count=2
    return count

def write_to_json(filename, writer_method, keys, values):
    num_keys = len(keys)
    assert(num_keys == len(values)), 'keys and values must have equal lengths'
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        with open(filename, writer_method) as outfile:
            for i in range(num_keys):
                key = keys[i]
                value = values[i]
                json.dump({key : value}, outfile)
                outfile.write('\n')
    finally:
        lock.release()
        
def randomize_classes(sort_task_index, seed, validator_sort_task, validator_nonsort_task=None):
    '''
    Description: randomly reassigns (swaps) half the classes (and data) of each validator to the other
    '''
    
    print('\nRandomizing Classes', flush=True)
    
    num_classes = np.sum(list(validator_sort_task.dataset.task_to_num_classes.values()))
    
    # random classes for task1
    np.random.seed(seed=seed)
    random_classes_task1 = np.random.choice(a=np.arange(num_classes), size=num_classes//2, replace=False, p=None)
    random_classes_task1 = np.sort(random_classes_task1)

    # random classes for task2 
    random_classes_task2 = [i for i in range(num_classes) if i not in random_classes_task1]   
    random_classes_task2 = np.array(random_classes_task2)
    
    # assign classes by sort and nonsort task
    if sort_task_index==0:
        random_classes_sort_task = random_classes_task1
        random_classes_nonsort_task = random_classes_task2
    else:
        random_classes_sort_task = random_classes_task2
        random_classes_nonsort_task = random_classes_task1
        
    # now gather samples for sort_task
    random_samples_sort_task = []
    for sample in validator_sort_task.dataset.samples:
        if sample[1] in random_classes_sort_task:
            random_samples_sort_task.append(sample)
    for sample in validator_nonsort_task.dataset.samples:
        if sample[1] in random_classes_sort_task:
            random_samples_sort_task.append(sample) 
              
    # now gather samples for nonsort_task
    random_samples_nonsort_task = []
    for sample in validator_sort_task.dataset.samples:
        if sample[1] in random_classes_nonsort_task:
            random_samples_nonsort_task.append(sample)
    for sample in validator_nonsort_task.dataset.samples:
        if sample[1] in random_classes_nonsort_task:
            random_samples_nonsort_task.append(sample)  
     
    # now modify/update the validators
    validator_sort_task.dataset.samples = random_samples_sort_task
    validator_nonsort_task.dataset.samples = random_samples_nonsort_task
    
    print('\nvalidator_sort_task:')
    print(validator_sort_task.dataset.samples[0])
    print(validator_sort_task.dataset.samples[-1])
    
    print('\nvalidator_nonsort_task:')
    print(validator_nonsort_task.dataset.samples[0])
    print(validator_nonsort_task.dataset.samples[-1])

def get_selected_units(lesions_filename, task, index):
    lesion_data = get_lesion_data(lesions_filename)
    key = os.path.join('selected_units','SORTEDBY_' + task, str(index))
    selected_units = np.array(lesion_data[key])
    return selected_units, key

def get_selected_losses(lesions_filename, task, index):
    lesion_data = get_lesion_data(lesions_filename)
    key = os.path.join('selected_losses','SORTEDBY_' + task, str(index))
    selected_units = np.array(lesion_data[key])
    return selected_units, key

def load_record(filename):
    '''
    Description:
        Since npz file are ovewritten by several jobs at similar times may run into 
        corrupted files by opening at the same time resulting in an error. By using a while try
        loop you can avoid this until the coast is clear to load. 
    Input:
        filename: filename with extension .npz
    Return:
        record: an npz record. To see contents try record.files
    '''
    file_not_loaded=True
    attempts=1
    while file_not_loaded:
        try:
            record=np.load(filename, allow_pickle=True)
            file_not_loaded=False
        except:
            print('\nWHILE LOADING FILE: Failed Attempt', attempts, '.\n', flush=True)
            #time.sleep(5)
            attempts+=1
    return record

def get_latest_npz_filename(dir):
    '''
    Description:
        Returns the latest file in the directory dir
    '''
    list_of_files = glob.glob(os.path.join(dir, '*.npz')) 
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
    except:
        latest_file = None
    return latest_file

def printFormat(name, var):
    print("{0:<40}: {1:}".format(name,var),flush=True)

def get_lesion_data(filename):
    '''Description:
            returns a dictionary of all json objects in the specificed jsonlines file
            object occuring more than once with same key will appear uniquely in returned dictionary 
            with the last json object overwriting previous json objects of the same key
        Returns:
            lesion_data: python dciionary
    '''
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        lesion_data = {}
        with jsonl.open(filename) as reader:
            for obj in reader:
                key = list(obj.keys())[0]
                lesion_data[key]=obj[key]
    finally:
        lock.release()
    return lesion_data

def get_predictions(filename):
    '''Description:
            returns a dictionary of all json objects in the specificed jsonlines file
            object occuring more than once with same key will appear uniquely in returned dictionary 
            with the last json object overwriting previous json objects of the same key
        Returns:
            lesion_data: python dciionary
    '''
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        predictions = {}
        with jsonl.open(filename) as reader:
            for obj in reader:
                key = list(obj.keys())[0]
                predictions[key]=obj[key]
    finally:
        lock.release()
        
    return predictions

def printLesionStatus(lesion_data):
    '''
    Description: 
        prints completion status for every layer submitted.
    '''
    keys = list(lesion_data.keys())
    keys.sort()
    for key in keys:
        if key.startswith('status'):
            value = lesion_data[key]
            printFormat(key, value)
    print('\nIndexes not shown have not completed.')

def json_completion_status(filename, sort_task, param_group_index):
    '''
    Description: 
        True is status for group_index lesion is complete, False otherwise
    Returns:
        is_complete - boolean
    '''
    lesion_data = get_lesion_data(filename)
    
    key = os.path.join('status', 'SORTEDBY_' + sort_task, str(param_group_index))
    value='not submitted'
    if key in lesion_data:
        value = lesion_data[key]
    is_complete=False
    if value == 'complete':
        is_complete=True
    return is_complete

def force_new_progress_record(selected_units, selected_losses, num_units, progress_dir):
    '''
    Descritption: Sometimes these files get corrupted and so here we force overwrite with new file'
    '''
    filename = get_latest_npz_filename(progress_dir)
    remaining_units = np.delete(arr=np.arange(num_units), obj=np.array(selected_units))
    remaining_units = np.delete(arr=remaining_units, obj=0) 
    np.savez(file            = filename,
             remaining_units = remaining_units, 
             dropped_units   = selected_units,
             dropped_losses  = selected_losses,
             next_iter_made  = np.array([False]),
             selection_made  = np.array([False]))
    return None
   
def generate_unit(filename, selected_units, num_units, progress_dir, next_iter=False, overwrite=False, iteration=0):
    '''
    Desription: picks the first unit from the remaining units if one exists and returns it, 
                along with an updated selected_units array only returned for convenience 
                of saving at end. If no remaining unit exists will return None with no update
    '''
   
    if filename is None or overwrite==True:
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
        # create file and lock right away
        filename = os.path.join(progress_dir,'progress_record_ITER_' + str(iteration)) + '.npz'
        #os.mknod(filename)
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            remaining_units = np.delete(arr=np.arange(num_units), obj=np.array(selected_units))
            unit = remaining_units[0]
            remaining_units = np.delete(arr=remaining_units, obj=0) 
            seed_value = random.randint(0,1000)
            print('calling random seed 1:', seed_value, flush=True)
            np.savez(file            = filename,
                     remaining_units = remaining_units, 
                     dropped_units   = np.array([]).astype(int),
                     dropped_losses  = np.array([]).astype(float),
                     next_iter_made  = np.array([False]),
                     selection_made  = np.array([False]),
                     seed            = np.array([seed_value]),
                     conclusion_count= np.array([0]))
        finally:
            lock.release()
    else:
        # lock file
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            # load data
            record = load_record(filename)

            # get saved units (exception handling starts the iter over from scratch)
            for file in record.files:
                try:
                    record[file]
                except BadZipFile:
                    print('\nWHILE GEN UNIT: Corrupted File. Restarting Iter.\n', flush=True)
                    basename = os.path.basename(filename)
                    iteration = int(basename.split('_')[3].strip('.npz'))
                    unit, seed_value, filename = generate_unit(filename=filename, 
                                                               selected_units=selected_units, 
                                                               num_units=num_units, 
                                                               progress_dir=progress_dir, 
                                                               overwrite=True, 
                                                               iteration=iteration)
                    return unit, seed_value, filename

            remaining_units = record['remaining_units'] 
            dropped_units   = record['dropped_units']
            dropped_losses  = record['dropped_losses']
            next_iter_made  = record['next_iter_made']
            selection_made  = record['selection_made']
            seed            = record['seed']
            conclusion_count= record['conclusion_count']
            

            # generate unit if there is one and overwrite file
            if remaining_units.shape[0] > 0:
                unit = remaining_units[0]
                seed_value = seed[0]
                print('retreiving stored seed 2:', seed_value, flush=True)
                remaining_units = np.delete(arr=remaining_units, obj=0)  # removes the zeroth index
                # filename unchanged
                np.savez(file            = filename,
                         remaining_units = remaining_units, 
                         dropped_units   = dropped_units,
                         dropped_losses  = dropped_losses,
                         next_iter_made  = next_iter_made,
                         selection_made  = selection_made,
                         seed            = seed,
                         conclusion_count = conclusion_count)
            elif remaining_units.shape[0] == 0 and next_iter == False:
                # filename unchanged
                unit = None
                seed_value = None
                print('None seed 3:', seed_value, flush=True)
            elif (next_iter_made[0] == False) and (remaining_units.shape[0] == 0) and (next_iter == True):
                unit = None
                seed_value = random.randint(0,1000)
                print('calling random seed 4:', seed_value, flush=True)
                print('selection_made:', selection_made[0], flush=True)
                next_remaining_units = np.delete(arr=np.arange(num_units), obj=np.array(selected_units))            
                basename = os.path.basename(filename)
                next_iteration = int(basename.split('_')[3].strip('.npz')) + 1 # retrieves the iteration and adds 1
                next_filename = os.path.join(progress_dir,'progress_record_ITER_' + str(next_iteration)) + '.npz'
                next_lockname = next_filename + '.lock'
                next_lock = FileLock(next_lockname, timeout=lock_timeout)
                next_lock.acquire(timeout=acquire_timeout)
                
                # update so that next iter is true
                np.savez(file             = filename,
                         remaining_units  = remaining_units, 
                         dropped_units    = dropped_units,
                         dropped_losses   = dropped_losses,
                         next_iter_made   = np.array([True]),
                         selection_made   = selection_made,
                         seed             = seed,
                         conclusion_count = conclusion_count )
                try:
                    # filename changed
                    filename=next_filename
                    np.savez(file             = next_filename,
                             remaining_units  = next_remaining_units, 
                             dropped_units    = np.array([]).astype(int),
                             dropped_losses   = np.array([]).astype(float),
                             next_iter_made   = np.array([False]),
                             selection_made   = np.array([False]),
                             seed             = np.array([seed_value]),
                             conclusion_count = np.array([0]) )
                finally:
                    next_lock.release()
            else:
                print('Entered make next_iter but has already been done!!!', flush=True)
                print('next_iter_made', next_iter_made[0], flush=True)
                unit=None
                seed_value=None
        finally:
            lock.release()
            
    return unit, seed_value, filename

def update_progress(filename, unit, loss):
    '''
    Description: appends the new unit and loss to the existing dropped units
    '''
    
    assert(unit is not None), 'unit must be an integer valued scalar'
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:

        # load data
        units_progress = load_record(filename)

        # get saved units
        remaining_units = units_progress['remaining_units']
        dropped_units = units_progress['dropped_units']
        dropped_losses = units_progress['dropped_losses']
        next_iter_made = units_progress['next_iter_made']
        selection_made = units_progress['selection_made']
        seed           = units_progress['seed']
        conclusion_count = units_progress['conclusion_count']

        # make update
        dropped_units = np.append(dropped_units, unit)
        dropped_losses = np.append(dropped_losses, loss)
        
        # make unique if necessary
        dropped_units, dropped_losses = make_unique(units=dropped_units, losses=dropped_losses)
        
        # overwrite file
        np.savez(file            = filename,
                 remaining_units = remaining_units, 
                 dropped_units   = dropped_units,
                 dropped_losses  = dropped_losses,
                 next_iter_made  = next_iter_made,
                 selection_made  = selection_made,
                 seed            = seed,
                 conclusion_count = conclusion_count)
    finally:
        lock.release()

    return None

def conclude_progress(filename):
    '''
    Completes the progress record
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
    
        # load data
        units_progress = load_record(filename)
        
        # get saved units
        remaining_units = units_progress['remaining_units']
        dropped_units = units_progress['dropped_units']
        dropped_losses = units_progress['dropped_losses']
        next_iter_made = units_progress['next_iter_made']
        selection_made = np.array([True])
        seed           = units_progress['seed']
        conclusion_count = np.array([units_progress['conclusion_count'][0] + 1])

        # overwrite file
        np.savez(file            = filename,
                 remaining_units = remaining_units, 
                 dropped_units   = dropped_units,
                 dropped_losses  = dropped_losses,
                 next_iter_made  = next_iter_made,
                 selection_made  = selection_made,
                 seed            = seed,
                 conclusion_count = conclusion_count)
    finally:
        lock.release()
    
    return conclusion_count[0]

def get_progress(filename):
    '''
    Description: retrieves the the dropped units and corresponding losses
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
    
        # load progress record
        units_progress = load_record(filename)

        # get saved units
        remaining_units = units_progress['remaining_units']
        dropped_units   = units_progress['dropped_units']
        dropped_losses  = units_progress['dropped_losses']
        next_iter_made  = units_progress['next_iter_made'][0]
        selection_made  = units_progress['selection_made'][0]
        num_remaining   = remaining_units.shape[0]
        seed_value      = units_progress['seed'][0]
        conclusion_count = units_progress['conclusion_count'][0]
        
    finally:
        lock.release()
    
    return dropped_units, dropped_losses, next_iter_made, selection_made, conclusion_count, num_remaining, seed_value

def get_selections(filename, selections_dir=None, overwrite=False):
    '''
    Description: returns selections, if filename is None then creates a new 'empty' record
    '''
    if filename is None or overwrite==True:   
        if not os.path.exists(selections_dir):
            os.makedirs(selections_dir)
            
        # create file and lock right away
        filename = os.path.join(selections_dir, 'selections_record.npz')
        #os.mknod(filename)
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            selected_units      = np.array([]).astype(int)
            selected_losses     = np.array([]).astype(int)
            selections_complete = np.array([False])

            np.savez(filename, 
                     selected_units=selected_units, 
                     selected_losses=selected_losses, 
                     selections_complete=selections_complete)
        finally:
            lock.release()
    else:
        #time.sleep(2)
        # load selections
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            selections          = load_record(filename)
            selected_units      = selections['selected_units']
            selected_losses     = selections['selected_losses']
            selections_complete = selections['selections_complete']
        finally:
            lock.release()
        
    return selected_units, selected_losses, selections_complete[0], filename

def update_selections(filename, selected_unit, selected_loss):
    '''
    Description: appends the new selections to existing selections
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        # load selections
        selections          = load_record(filename)
        selected_units      = selections['selected_units']
        selected_losses     = selections['selected_losses']
        selections_complete = selections['selections_complete']

        # update selections
        selected_units = np.append(selected_units, selected_unit)
        selected_losses = np.append(selected_losses, selected_loss)

        # overwrite with new udpates
        #np.savez(filename, selected_units=selected_units, selected_losses=selected_losses)
        np.savez(filename, 
                 selected_units=selected_units, 
                 selected_losses=selected_losses, 
                 selections_complete=selections_complete)
    finally:
        lock.release()
    
    return None

def conclude_selections(filename):
    '''
    Description: concludes the selections record
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
    
        # load selections
        selections          = load_record(filename)
        selected_units      = selections['selected_units']
        selected_losses     = selections['selected_losses']

        # conclude
        selections_complete = np.array([True])

        # overwrite file
        np.savez(filename, 
                 selected_units=selected_units, 
                 selected_losses=selected_losses, 
                 selections_complete=selections_complete)
    finally:
        lock.release()
        
    return None

def get_drop_loss(selected_units, candidate_unit, validator, weight, bias, cache, ngpus, max_batches, seed):
    
    # drop units 
    # -------------------------
    drop_units = np.append(selected_units,candidate_unit)
    new_weight = weight.clone()
    new_bias = bias.clone()
    for unit in drop_units:
        new_weight[unit] = 0.0
        if bias is not None:
            new_bias[unit]   = 0.0
        
    # get loss on prediction
    # -------------------------
    _, _, _, _, loss, _ = helper.predict(model=validator.model, 
                                         data_loader=validator.data_loader, 
                                         ngpus=ngpus, 
                                         topk=1,
                                         max_batches=max_batches,
                                         reduce_loss=True,
                                         notebook=False,
                                         seed=seed)
    
    # replace unit 
    # -------------------------
    for unit in drop_units:
        new_weight[unit] = torch.from_numpy(cache['W'][unit])
        if bias is not None:
            new_bias[unit]   = torch.from_numpy(cache['b'][unit]) 
    
    return loss

def restore_missing_units(num_units, progress_record_filename, selections_record_filename, restore=False):
    '''
    Description: sometimes all units are not returned/skipped and waiting for jobs to return their units gets stuck 
    in an infinite loop. To avoid this the finally job has the option to restore any missing units in order to not wait forever.
    This function creates this functionality. It looks through the dropped_units and if there are any missing, it will restore
    these to the remaining units and continue lesioning. 
    '''
    progress_record = load_record(filename=progress_record_filename)
    
    # get saved units
    remaining_units  = progress_record['remaining_units']
    dropped_units    = progress_record['dropped_units']
    dropped_losses   = progress_record['dropped_losses']
    next_iter_made   = progress_record['next_iter_made']
    selection_made   = progress_record['selection_made']
    seed_value       = progress_record['seed']
    conclusion_count = progress_record['conclusion_count']
    
    print(len(dropped_units))
    print(len(dropped_losses))
    
    # make sure you restore missing ones but also remove anything that is duplicate
    dropped_units, dropped_losses = make_unique(units=dropped_units, losses=dropped_losses)
    
    print(len(dropped_units))
    print(len(dropped_losses))

    # print current
    print('\nBefore Restoring Missing Units:',flush=True)
    print('-------------------------------',flush=True)
    print('remaining_units:', len(remaining_units),flush=True)
    print('num dropped_units:', len(dropped_units),flush=True)
    print('num dropped_losses:', len(dropped_losses),flush=True)
    print('next_iter_made:', next_iter_made,flush=True)
    print('selection_made:', selection_made,flush=True)
    print('seed_value:', seed_value,flush=True)
    print('conclusion_count:', conclusion_count,flush=True)
    print(flush=True)
    
    # get missed units
    missed_units = []
    for i in range(num_units):
        if i not in dropped_units:
            missed_units.append(i)
    missed_units = np.array(missed_units).astype(int)
    conclusion_count = np.array([0]).astype(int)
    selection_made = np.array([False])
    next_iter_made = np.array([False])
    print('missing_units:', missed_units, flush=True)
    print(flush=True)
    
    if restore:
        # progress record
        lockname = progress_record_filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            np.savez(file             = progress_record_filename,
                     remaining_units  = missed_units, 
                     dropped_units    = dropped_units,
                     dropped_losses   = dropped_losses,
                     next_iter_made   = next_iter_made,
                     selection_made   = selection_made,
                     seed             = seed_value,
                     conclusion_count = conclusion_count)
        finally:
            lock.release()
        
        # selections record ## THIS IS PROBABLY UNECESSARY
        lockname = selections_record_filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            np.savez(file                 = selections_record_filename,
                     selected_units       = np.array([]).astype(int), 
                     selected_losses      = np.array([]).astype(int),
                     selections_complete  = np.array([False]))
        finally:
            lock.release()

        
        ## get saved units
        progress_record = load_record(filename=progress_record_filename)
        remaining_units  = progress_record['remaining_units']
        dropped_units    = progress_record['dropped_units']
        dropped_losses   = progress_record['dropped_losses']
        next_iter_made   = progress_record['next_iter_made']
        selection_made   = progress_record['selection_made']
        seed_value       = progress_record['seed']
        conclusion_count = progress_record['conclusion_count']

        # print current
        print('\nAfter Restoring Missing Units:')
        print('-------------------------------',flush=True)
        print('remaining_units:', len(remaining_units))
        print('dropped_units:', len(dropped_units))
        print('dropped_losses:', len(dropped_losses))
        print('next_iter_made:', next_iter_made)
        print('selection_made:', selection_made)
        print('seed_value:', seed_value)
        print('conclusion_count:', conclusion_count)
        
    
    return None

   
def greedy_lesion_layer(validator, index, layerMap, selections_dir, progress_dir, sort_task, lesions_filename, iterator_seed, 
                        p=0.0, ngpus=0, max_batches=None, approx_method=None):
    '''
    Description:
        Runs greedy lesion on a specified layer. 
    Inputs:
        validator - validator object, see utils
        index     - index of parameter (so 0 will be 0th pair of weight and bias, and so on)
        layerMap  - layerMap object, see utils
        p         - percent of units to select
        
    '''
    
    print('Starting Greedy Layer Lesion on Train Data',flush=True)
    print('------------------------------------------',flush=True)
    print()
    
    print('Using Methods:')
    print('Greedy @ p=',p)
    print('Approximation Method -', approx_method)
    print()
    
    cache = {'W':{}, 'b':{}}
    weight, bias = helper.getWeightandBias(validator.model, layerMap, index)
    num_units = weight.shape[0]
    layer = layerMap['ParamGroupIndex2LayerIndex'][index]
    layerType = layerMap['ParamGroupIndex2LayerType'][index]
    print("{0:<40}: {1:}".format('validator.name',validator.name),flush=True)
    print("{0:<40}: {1:}".format('index',index),flush=True)
    print("{0:<40}: {1:}".format('layer',layer),flush=True)
    print("{0:<40}: {1:}".format('layerType',layerType),flush=True)
    print("{0:<40}: {1:}".format('num_units',num_units),flush=True)
    num_units_to_drop = np.round(num_units*p).astype(int)
    if p==0.0:
        num_units_to_drop=num_units
    print("{0:<40}: {1:}".format('num_units_to_drop',num_units_to_drop),flush=True)
    print(flush=True)
    # Base Loss
    # -------------------------
    _, _, _, _, loss, _ = helper.predict(model=validator.model, 
                                         data_loader=validator.data_loader, 
                                         ngpus=ngpus, 
                                         topk=1,
                                         max_batches=max_batches,
                                         reduce_loss=True)
    
    print("{0:<40}: {1:}".format('loss @base',loss),flush=True)
    print(flush=True)
        
    # cache all units
    # -------------------------------------------------------
    for unit in range(num_units):
        cache['W'][unit] = weight[unit].detach().cpu().numpy()
        if bias is not None:
            cache['b'][unit] = bias[unit].detach().cpu().numpy()
    
    # run until have reached the num_units required to select
    # -------------------------------------------------------
    latest_selections_record = get_latest_npz_filename(dir=selections_dir) # latest filename of selections record (or None)
    selected_units, selected_losses, _, latest_selections_record = get_selections(filename=latest_selections_record,
                                                                                  selections_dir=selections_dir)
    
    # run until greedy has (p*100)% of units selected (if p=0 a break statement will occur)
    linear_complete=False
    greedy_complete=False
    while (selected_units.shape[0] <= num_units_to_drop): 
        print(flush=True)
        print(flush=True)
        print('---------------------------------------------------------------------',flush=True)
        print('/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/',flush=True) 
        latest_progress_record = get_latest_npz_filename(dir=progress_dir) 
        print('latest_progress_record:', latest_progress_record, flush=True)
        print(flush=True)
        print('selected_units:', selected_units,flush=True)
        print(flush=True)
        print('selected_losses:', selected_losses,flush=True)
        print(flush=True)
        print('Losses conditioned on ' + str(len(selected_units)) + ' selected units.',flush=True) 
        print('---------------------------------------------------------------------',flush=True) 
      
        # run until a break statement (breaks out of nested while loop)
        while True:
            #latest_progress_record = get_latest_npz_filename(dir=progress_dir)
            unit, seed_value, latest_progress_record = generate_unit(filename=latest_progress_record, 
                                                                     selected_units=selected_units, 
                                                                     num_units=num_units, 
                                                                     progress_dir=progress_dir)
            if unit is None:
                print('Unit == None', flush=True)
                break
            
            if iterator_seed is None:
                seed_value = None
            elif iterator_seed == "fixed":
                seed_value = fixed_seed_value
            elif iterator_seed == "selection":
                pass # keep the seed_value from previous generate_unit call
                
            loss = get_drop_loss(selected_units=selected_units, 
                                 candidate_unit=unit, 
                                 validator=validator, 
                                 weight=weight,
                                 bias=bias,
                                 cache=cache,
                                 ngpus=ngpus, 
                                 max_batches=max_batches,
                                 seed=seed_value) 
            
            print("{0:<40}: {1:}".format('seed @unit ' + str(unit),seed_value),flush=True)
            print("{0:<40}: {1:}".format('loss @unit ' + str(unit),loss),flush=True)
            
            update_progress(filename=latest_progress_record, unit=unit, loss=loss)
        print('\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/',flush=True) 
        print('--------------------------------------------------------------------',flush=True)
        print(flush=True)
        print(flush=True)
        
        # should force loop until all candidates have been computed
        progress = get_progress(filename=latest_progress_record)
        candidate_units, candidate_losses, next_iter_made, selection_made, _, num_remaining, _ = progress
        if num_remaining > 0:
            print('num_remaining', num_remaining, flush=True)
            print('selection_made', selection_made, flush=True)
            selected_units, selected_losses, _, _ = get_selections(filename=latest_selections_record)
            continue
       
        # make selection 
        # -------------------------
        # fix
        #progress = get_progress(filename=latest_progress_record)
        #candidate_units, candidate_losses, next_iter_made, selection_made, num_remaining, _ = progress
        print('selection_made test:', selection_made)
        print('next_iter_made test:', next_iter_made)
        if (selection_made==False):
            
            conclusion_count = conclude_progress(filename=latest_progress_record) # confirm to avoid repeating
            print('conclusion_count test:', conclusion_count)
            
            if (conclusion_count > 1):
                print('ignoring conclusion test')
                time.sleep(30)
                continue
                        
            print('\nNumber of Selection Candidates:', len(candidate_units),flush=True) # erase
            print(flush=True)
            if p>0:
                while len(np.union1d(selected_units, candidate_units)) < num_units:
                    print('\nWaiting for all selection candidates...', flush=True)
                    print('num selected units:', len(selected_units),flush=True)
                    print('num candidate units:', len(candidate_units), flush=True)
                    print('num of total units:', len(np.union1d(selected_units, candidate_units)), flush=True)
                    time.sleep(10)
                    progress = get_progress(filename=latest_progress_record)
                    candidate_units, candidate_losses, _, _, _, _, _ = progress
                print('All candidates complete!\n', flush=True)
                
                argmax = np.argmax(candidate_losses)
                selected_unit = candidate_units[argmax]
                selected_loss = candidate_losses[argmax]
              
                update_selections(filename=latest_selections_record, 
                                  selected_unit=selected_unit, 
                                  selected_loss=selected_loss)
                
                selected_units, selected_losses, _, _ = get_selections(filename=latest_selections_record,
                                                                       selections_dir=selections_dir)   

                _ = generate_unit(filename=latest_progress_record,  
                                  selected_units=selected_units, 
                                  num_units=num_units, 
                                  progress_dir=progress_dir,
                                  next_iter=True)
            elif p==0.0:
                num_candidates = len(candidate_units)
                waiting_time=0.0
                while num_candidates < num_units:
                    print('\nWaiting for all selection candidates...', flush=True)
                    time.sleep(30)
                    waiting_time+=10.0
                    progress = get_progress(filename=latest_progress_record)
                    candidate_units, candidate_losses, _, _, _, _, _ = progress
                    num_candidates = len(candidate_units)
                    print('Num candidates:', num_candidates)
                    
                    if (waiting_time > 60.0*35):
                        print('\nRestoring Missing Units...\n', flush=True)
                        restore_missing_units(num_units=num_units, 
                                              progress_record_filename=latest_progress_record, 
                                              selections_record_filename=latest_selections_record, 
                                              restore=True)
                        continue
                    
                print('All candidates complete!\n', flush=True)
                
                sorted_candidates = np.argsort(candidate_losses)[::-1]
                selected_units, selected_losses = candidate_units[sorted_candidates], candidate_losses[sorted_candidates]
                
                update_selections(filename=latest_selections_record, 
                                  selected_unit=selected_units, 
                                  selected_loss=selected_losses)
                
                linear_complete=True
                break
             
        # if selection has been made and p==0
        elif p==0:
            break
        # if selection has been made and p>0
        
        else: # (num_remaining==0 and selection_made==True)
            # case 1: just waint until new record it readable
            _, _, next_iter_made, _, _, _, _ = get_progress(filename=latest_progress_record)
            if next_iter_made==True:
                latest_progress_record_temp = get_latest_npz_filename(dir=progress_dir)
                while (latest_progress_record == latest_progress_record_temp):
                    time.sleep(15)
                    print('Waiting...', flush=True)
                    latest_progress_record_temp = get_latest_npz_filename(dir=progress_dir)
                
                selected_units, selected_losses, _, _ = get_selections(filename=latest_selections_record,
                                                                       selections_dir=selections_dir)
                continue
            # case 2: wait until next_iter_made==True or make a new one
            else:
                total_wait = 0
                while next_iter_made==False:
                    print('Waiting for selection to be made...', flush=True)
                    time.sleep(15)
                    total_wait+=15

                    if total_wait > 15*20:
                        print('Next Iter Not Found. Creating new progress record.')
                        '''
                        selected_units, selected_losses, _, _ = get_selections(filename=latest_selections_record)
                        latest_progress_record_temp = get_latest_npz_filename(dir=progress_dir)
                        if (selection_made==True) and (latest_progress_record==latest_progress_record_temp) and (num_remaining==0):
                        _ = generate_unit(filename=latest_progress_record, 
                                          selected_units=selected_units, 
                                          num_units=num_units, 
                                          progress_dir=progress_dir,
                                          next_iter=True)
                        '''
                    _, _, next_iter_made, _, _, _, _ = get_progress(filename=latest_progress_record)

                print('\nWaiting time:', total_wait, 'seconds', flush=True)
                continue 
            
            '''
            time.sleep(5)
            _, _, next_iter_made, _, _, _, _ = get_progress(filename=latest_progress_record)
            if next_iter_made:
                print('next_iter_made test final:', next_iter_made, flush=True)
                continue
            else: # next_iter_made==False
            
                #print('selection_made_prev', selection_made)
                print('Sleep Statement, Selection Made and Num Remaining is Zero', flush=True)
                time.sleep(10)


                _,_, next_iter_made, selection_made, _, num_remaining, _ = get_progress(filename=latest_progress_record) 
                selected_units, selected_losses, _, _ = get_selections(filename=latest_selections_record)
                latest_progress_record_temp = get_latest_npz_filename(dir=progress_dir) 

                #print('selection_made', selection_made, flush=True)
                #print('next_iter_made', next_iter_made, flush=True)
                #print('latest_progress_record_temp', latest_progress_record_temp, flush=True)
                #print('num_remaining', num_remaining, flush=True)
                #print(flush=True)

                # only required in case a job was canceled before creating a new file

                if (selection_made==True) and (latest_progress_record==latest_progress_record_temp) and (num_remaining==0):
                    _ = generate_unit(filename=latest_progress_record, 
                                      selected_units=selected_units, 
                                      num_units=num_units, 
                                      progress_dir=progress_dir,
                                      next_iter=True)
                continue
                '''
       
    if p>0.0:
        selected_units, selected_losses, selections_complete, _ = get_selections(latest_selections_record)
        greedy_complete=(selected_units.shape[0] > num_units_to_drop)
        if greedy_complete and selections_complete==False:
            conclude_selections(filename=latest_selections_record)
    else:
        selected_units, selected_losses, selections_complete, _ = get_selections(latest_selections_record)
        if linear_complete and selections_complete==False:
            conclude_selections(filename=latest_selections_record)
        
    print(flush=True)
    print('selected_units:', selected_units,flush=True)
    print('selected_losses:', selected_losses,flush=True)
    print(flush=True)
    return None




    
def run_lesion():
    # FLAGS 
    # ------------------------------------------------------
    parser = argparse.ArgumentParser(description='Lesion Filters')
    parser.add_argument('--config_file',          default=None,        type=str,  help='path to config file')
    parser.add_argument('--param_group_index',    default=0,           type=int,  help='param weight and bias group index')
    parser.add_argument('-p', '--greedy_p',       default=0.0, type=float,  help='percent greedy iterations, 0.0==disabled')
    parser.add_argument('--shuffle',              default=False,       type=bool, help='shuffle data in dataloader')
    parser.add_argument('--ngpus',                default=1,           type=int,  help='number of gpus to use')
    parser.add_argument('--batch_size',           default=128,         type=int,  help='batch size')
    parser.add_argument('--max_batches',          default=5,           type=int,  help='batches to run on train losses')
    parser.add_argument('--workers',              default=1,           type=int,  help='read and write workers')
    parser.add_argument('--sort_task_index',      default=0,           type=int,  help='sort_task_index + 1')
    parser.add_argument('--nonsort_task_index',   default=1,           type=int,  help='nonsort_task_index + 1')
    parser.add_argument('--restore_epoch',        default=-1,          type=int,  help='epoch to restore from')
    parser.add_argument('--lesion_name',          default='',          type=str,  help='name of lesion')
    parser.add_argument('--lesions_dir',          default='./lesions/',type=str,  help='where to read the losses from')
    parser.add_argument('--iterator_seed', choices=['fixed', 'selection', None], default=None, type=str, help='seed type')
    parser.add_argument('--read_seed',            default=None,        type=int,  help='seed type')
    parser.add_argument('--maxout',               default=False,       type=bool, help='read all data and then shuffle')
    parser.add_argument('--randomize_classes',     default=False,      type=bool, help='wether to randomly mix classes')
    parser.add_argument('--randomize_classes_seed', default=0,         type=int, help='how to mix the classes')
    
    
    FLAGS, FIRE_FLAGS = parser.parse_known_args()
    
    if FLAGS.ngpus > 0:
        torch.backends.cudnn.benchmark = True
              
    # Get Model and tasks
    # --------------------------
    config = helper.Config(config_file=FLAGS.config_file)
    model, ckpt_data = config.get_model(pretrained=True, ngpus=FLAGS.ngpus, dataParallel=True, epoch=FLAGS.restore_epoch)
    layerMap = helper.getLayerMapping(model)
        
    config.batch_size = FLAGS.batch_size
    tasks = list(config.max_valid_samples.keys())
    tasks.sort()
    sort_task = tasks[FLAGS.sort_task_index]
    nonsort_task = tasks[FLAGS.nonsort_task_index]
    
    ###########################################################################
    ###########################################################################
    #                               LESION                                    #
    ###########################################################################
    ###########################################################################
    
    # validator sort_task train data
    # --------------------------
    config_sort_task_train_data = helper.Config(config_file=FLAGS.config_file)
    max_train_samples = copy.deepcopy(config_sort_task_train_data.max_train_samples)
    max_train_samples[nonsort_task] = 0
    validator_sort_task_train_data = helper.Validator(name='sort_task_train_data', 
                                                      model=model, 
                                                      batch_size=FLAGS.batch_size,
                                                      data_dir=config_sort_task_train_data.data_dir, 
                                                      data_subdir='train',
                                                      max_samples=max_train_samples,
                                                      maxout=FLAGS.maxout,
                                                      read_seed=FLAGS.read_seed,
                                                      ngpus=FLAGS.ngpus, 
                                                      shuffle=FLAGS.shuffle,
                                                      includePaths=True,
                                                      workers=FLAGS.workers)
    
    if FLAGS.randomize_classes:
        config_sort_task_train_data = helper.Config(config_file=FLAGS.config_file)
        max_train_samples = copy.deepcopy(config_sort_task_train_data.max_train_samples)
        max_train_samples[sort_task] = 0
        validator_nonsort_task_train_data = helper.Validator(name='nonsort_task_train_data', 
                                                          model=model, 
                                                          batch_size=FLAGS.batch_size,
                                                          data_dir=config_sort_task_train_data.data_dir, 
                                                          data_subdir='train',
                                                          max_samples=max_train_samples,
                                                          maxout=FLAGS.maxout,
                                                          read_seed=FLAGS.read_seed,
                                                          ngpus=FLAGS.ngpus, 
                                                          shuffle=FLAGS.shuffle,
                                                          includePaths=True,
                                                          workers=FLAGS.workers)
        
        randomize_classes(sort_task_index        =FLAGS.sort_task_index, 
                          seed                   =FLAGS.randomize_classes_seed, 
                          validator_sort_task    =validator_sort_task_train_data, 
                          validator_nonsort_task =validator_nonsort_task_train_data)
        del validator_nonsort_task_train_data
        
        
    # print arguments
    # --------------------------
    config.printAttributes()
    print(flush=True)
    print('--------MODEL-----------',flush=True)
    print('------------------------',flush=True)
    print(model,flush=True)
    print('------------------------',flush=True)
    print(flush=True)
    
    helper.printArguments(config=config_sort_task_train_data, 
                          validator=validator_sort_task_train_data, 
                          mode='train', 
                          FLAGS=FLAGS)
    
    # Save File (HDF5)
    # --------------------------    
    print('-------SAVE FILE--------',flush=True)
    print('------------------------',flush=True)
    network_dir = os.path.basename(os.path.dirname(FLAGS.config_file))
    lesions_dir = os.path.join(FLAGS.lesions_dir, network_dir, config.name)
    lesions_filename = 'lesions_LESION_NAME_' + FLAGS.lesion_name 
    predictions_filename = 'predictions_LESION_NAME_' + FLAGS.lesion_name 
    lesions_filename = os.path.join(lesions_dir, lesions_filename) + '.jsonl'
    predictions_filename = os.path.join(lesions_dir, predictions_filename) + '.jsonl'
    
    print(flush=True)
    print('Results being saved to:', lesions_filename,flush=True)
    print(flush=True)
    
    task_to_sort_by = sort_task
    if FLAGS.randomize_classes:
        task_to_sort_by = 'randomizedClasses_task_' + str(FLAGS.sort_task_index) + '_seed_' + str(FLAGS.randomize_classes_seed)
    records_dir = ['LESION_NAME_'+ FLAGS.lesion_name,
                    'SORTEDBY_' + task_to_sort_by, 
                    'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index)]
    records_dir = '_'.join(records_dir)
    
    selections_dir = os.path.join(lesions_dir, 'selections_records', records_dir)
    progress_dir   = os.path.join(lesions_dir, 'progress_records',   records_dir)
            
    print(flush=True)
    print('Record Files:', flush=True)
    print('\nSelections Records:', selections_dir,flush=True)
    print('\nProgress Records:', progress_dir,flush=True)
    print(flush=True)
                                                        
    if os.path.isfile(lesions_filename):
        print('Adding to existing jsonlines file...',flush=True)
        print(flush=True)
    else:
        print('Creating new jsonlines file...',flush=True)
        print(flush=True)
        if not os.path.exists(lesions_dir):
            os.makedirs(lesions_dir)
        
        writer_method = 'w'
        keys = ['meta/greedy_p', 'meta/shuffle', 'meta/batch_size', 'meta/max_batches', 'meta/restore_epoch']
        values = [FLAGS.greedy_p, FLAGS.shuffle, FLAGS.batch_size, FLAGS.max_batches, FLAGS.restore_epoch]
        write_to_json(filename=lesions_filename, writer_method=writer_method, keys=keys, values=values)
    
    writer_method = 'a' # add to json file from now on
    
    # get latest selections
    latest_selections_record     = get_latest_npz_filename(dir=selections_dir) 
    selections_complete=False
    if latest_selections_record is not None:
        _, _, selections_complete, _ = get_selections(filename=latest_selections_record, 
                                                      selections_dir=selections_dir)
    
    if (latest_selections_record is None) or (selections_complete==False): 
         
        greedy_lesion_layer(validator=validator_sort_task_train_data, 
                            index=FLAGS.param_group_index,
                            layerMap=layerMap, 
                            selections_dir=selections_dir,
                            progress_dir=progress_dir,
                            sort_task=task_to_sort_by,
                            lesions_filename=lesions_filename,
                            iterator_seed=FLAGS.iterator_seed,
                            p=FLAGS.greedy_p, 
                            ngpus=FLAGS.ngpus, 
                            max_batches=FLAGS.max_batches)

    # wait for all processes to finish 
    while selections_complete==False: 
        latest_selections_record = get_latest_npz_filename(dir=selections_dir)
        selections = get_selections(filename=latest_selections_record)
        selected_units, selected_losses, selections_complete, _ = selections
        print('\nWaiting on other jobs to complete...', flush=True)
        time.sleep(10)

    # search json file for complete status
    # -------------------------
    status_is_complete = json_completion_status(filename=lesions_filename, 
                                                sort_task=task_to_sort_by, 
                                                param_group_index=FLAGS.param_group_index)
    if (status_is_complete == False):
        json_conclude_count = conclude_lesion_to_json(filename=lesions_filename, 
                                                      sort_task=task_to_sort_by, 
                                                      param_group_index=FLAGS.param_group_index)
        
        print('\njson_conclude_count:', json_conclude_count, flush=True)
        
        if json_conclude_count > 1:
            print('\nResults being written to JSON by another job!', flush=True)
            time.sleep(30)
            return None
                
        # write selections to json
        print('\nWriting lesion results to JSON file!', flush=True)
        print(lesions_filename, flush=True)
        keys = []
        values = []
        selected_units, selected_losses, _, _ = get_selections(filename=latest_selections_record, selections_dir=selections_dir)
        
        group = os.path.join( 'selected_units', 'SORTEDBY_' + task_to_sort_by, str(FLAGS.param_group_index) )
        keys.append(group)
        values.append(selected_units.tolist())
        
        group = os.path.join('selected_losses', 'SORTEDBY_' + task_to_sort_by, str(FLAGS.param_group_index) )
        keys.append(group)
        values.append(selected_losses.tolist())
        
        write_to_json(filename=lesions_filename, writer_method=writer_method, keys=keys, values=values)
            

    time.sleep(30)
          
    print('\n---Lesion complete for current index!', flush=True)
    return None
    

if __name__ == "__main__": 
    run_lesion()
    print(flush=True)
    print('Lesion on Layer Complete.',flush=True)
    print(flush=True)
    
    



    
    


