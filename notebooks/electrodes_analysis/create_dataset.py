import numpy as np
import torch

def create_tensor_datasets(trials_dataset, true_labels, trials_sum_new):
    Dataset = torch.utils.data.TensorDataset(torch.permute(torch.Tensor(trials_dataset),(1,0,2)), torch.Tensor(true_labels))
    train_num = int(0.8*trials_sum_new)
    test_num = int(trials_sum_new-train_num)
    train_data, test_data = torch.utils.data.random_split(Dataset, [train_num, test_num])
    del Dataset
    return train_data, test_data

def filter_dataset_by_conditions(trials_dataset, true_labels, cues, primes, subjects_through_trials, simple = False, congruent_vs_incongruent = False, free_choice_vs_instruction = False, remove_subjects_groups = ['MULTI_EMG'], clean_artifacts = False):
    '''
    Filter the dataset by the conditions.

    simple: take only the RRR, LLL, NRR, NLLL trials
    congruent_vs_incongruent: change the lables such congruent trials are labeled 1 and incongruent trials are labeled 2
    free_choice_vs_instruction: change the lables such free choice trials are labeled 1 and instruction trials are labeled 2
    remove_subjects_groups: remove subjects groups that are given in the list


    '''
    if simple:
        condition = np.logical_or(np.logical_and(np.equal(primes, cues), np.equal(cues, true_labels)),np.logical_and(primes == 3, np.equal(cues, true_labels)))#np.logical_and(np.equal(primes, cues), np.equal(cues, true_labels))
        simple_indices = np.where(condition)[0]
        trials_dataset = trials_dataset[:,simple_indices,:]
        true_labels = true_labels[simple_indices]
        primes = primes[simple_indices]
        cues = cues[simple_indices]
        subjects_through_trials = subjects_through_trials[simple_indices]
        trials_sum_new = len(simple_indices)
    elif congruent_vs_incongruent:
        condition = primes != 3 
        trials_dataset = trials_dataset[:,condition,:]
        congruent_condition = np.logical_and(np.equal(primes, cues), np.equal(cues, true_labels))
        congruent_indices = np.where(congruent_condition)[0]
        true_labels[congruent_indices] = 1
        incongruent_condition = np.logical_and(np.logical_and(np.not_equal(primes, cues), np.equal(cues, true_labels)), np.not_equal(primes, 3*np.ones(len(primes))))
        incongruent_indices = np.where(incongruent_condition)[0]
        true_labels[incongruent_indices] = 2
        true_labels = true_labels[condition]
        primes = primes[condition]
        cues = cues[condition]
        subjects_through_trials = subjects_through_trials[condition]
        trials_sum_new = sum(condition)
    elif free_choice_vs_instruction:
        free_choice_condition = cues == 4
        free_choice_indices = np.where(free_choice_condition)[0]
        instruction_condition = cues != 4
        instruction_indices = np.where(instruction_condition)[0]
        true_labels[free_choice_indices] = 1
        true_labels[instruction_indices] = 2
    else:
        trials_sum_new = trials_dataset.shape[1]

    if clean_artifacts:
        clean_indices = filter_multi_emg_artifacts(trials_dataset, true_labels, cues, primes, subjects_through_trials)
        trials_dataset = trials_dataset[:, clean_indices, :]
        true_labels = true_labels[clean_indices]
        primes = primes[clean_indices]
        cues = cues[clean_indices]
        subjects_through_trials = subjects_through_trials[clean_indices]
        trials_sum_new = len(clean_indices)

    if len(remove_subjects_groups) > 0:
        relevant_subjects = {'YOUNG_1':[331,333,334,342,345,346,347,349],'YOUNG_2':[224,225,226,228,229,230,231,232,235,236,238,239,240,243],'OLDER_1':[302,303,304,306,308,309,313,320],'OLDER_2':[316,317,318,319,321,322,323,324,330],'MULTI_EMG':[506,508,509,510,511,513,514,515,516,517,518,519,520,521,522,523]}
        for group in remove_subjects_groups:
            subjects_group_ids = []
            for subject in relevant_subjects[group]:
                subjects_group_ids.append(subject_to_subject_id_mapping(relevant_subjects).subject_to_subject_id(subject))
        extract_group_indices = np.where(~np.isin(subjects_through_trials, subjects_group_ids))[0]
        trials_dataset = trials_dataset[:,extract_group_indices,:]
        true_labels = true_labels[extract_group_indices]
        primes = primes[extract_group_indices]
        cues = cues[extract_group_indices]
        subjects_through_trials = subjects_through_trials[extract_group_indices]
        trials_sum_new = len(extract_group_indices)

    return trials_dataset, true_labels, cues, primes, subjects_through_trials, trials_sum_new

def preprocessing_steges(trials_dataset, true_labels, cues, primes, subjects_through_trials, trials_sum_new, electrodes_names, electrodes_num = 64, normalize = True, cut_beginning = False, cut_end = False, without_short_trials = True, without_long_trials = True, specific_length_range = False, without_outliers = False, min_range_value=40, max_range_value =45, choose_electrodes = False, seperate_length = False, concatenate_details_to_data = False):
    '''
    Make preprocessing steps on the dataset.

    normalize: normalize the dataset
    cut_beginning: cut the first X ms from the trials
    cut_end: cut the last Y ms from the trials
    without_short_trials: remove trials less than 400 ms
    without_long_trials: remove trials longer than 800 ms
    specific_length_range: take only trials with specific length range
    choose_electrodes: take only the electrodes in the sub_electrodes_list
    seperate_length: cut specific time range from the start and end of trials, specifically to each trial length
    csp: apply common spatial patterns
    '''
    nan_mask = np.isnan(trials_dataset)
    non_nan_counts = np.sum(~nan_mask, axis=2)
    unrelevant_indices = []

    if without_short_trials:
        lengthes = np.unique(non_nan_counts)
        short_trials_mask = lengthes < 40
        short_lengths = lengthes[short_trials_mask]
        short_indices = np.where(np.isin(non_nan_counts[0, :], short_lengths))[0]
        unrelevant_indices.extend(short_indices)

    if without_long_trials:
        lengthes = np.unique(non_nan_counts)
        long_trials_mask = lengthes >= 80
        long_lengths = lengthes[long_trials_mask]
        long_indices = np.where(np.isin(non_nan_counts[0, :], long_lengths))[0]
        unrelevant_indices.extend(long_indices)

    if specific_length_range:
        outside_specific_length_indices = np.where(
            (non_nan_counts[0, :] < min_range_value) | (non_nan_counts[0, :] > max_range_value)
        )[0]
        unrelevant_indices.extend(outside_specific_length_indices)

    if without_outliers:
        outlier_indices = []
        for trial_idx in range(trials_dataset.shape[1]):
            trial_data = trials_dataset[:, trial_idx, :]
            if np.nanmax(np.abs(trial_data)) > 300:
                outlier_indices.append(trial_idx)
        unrelevant_indices.extend(outlier_indices)

    # Remove duplicates and sort
    unrelevant_indices = np.unique(unrelevant_indices)

    # Get relevant indices
    all_indices = np.arange(non_nan_counts.shape[1])
    relevant_indices = np.setdiff1d(all_indices, unrelevant_indices)
    # trials_dataset = trials_dataset[:,relevant_indices,:,:] 
    trials_dataset = trials_dataset[:,relevant_indices,:]  # Assuming trials_dataset is 3D (electrodes, trials, time)
    true_labels = true_labels[relevant_indices]
    primes = primes[relevant_indices]
    cues = cues[relevant_indices]
    subjects_through_trials = subjects_through_trials[relevant_indices]
    nan_mask = np.isnan(trials_dataset)
    non_nan_counts = np.sum(~nan_mask, axis=2) 
    trials_sum_new = trials_dataset.shape[1] 
    
    if normalize:
        # nan_mask = np.isnan(trials_dataset)
        # non_nan_counts = np.sum(~nan_mask, axis=2)
        # lengthes, labels = np.unique(non_nan_counts, return_inverse=True)
        # labelled_array = labels.reshape(non_nan_counts.shape) + 1
        # for i in range(len(lengthes)):
        #     for electrode in range(electrodes_num):
        #         trials_dataset[electrode,labelled_array[electrode]==i+1,:] = (trials_dataset[electrode,labelled_array[electrode]==i+1,:]-np.nanmean(trials_dataset[electrode,labelled_array[electrode]==i+1,:], axis = 0))/np.nanstd(trials_dataset[electrode,labelled_array[electrode]==i+1,:], axis = 0)
        subjects = np.unique(subjects_through_trials)
        for i in range(len(subjects)):
            for electrode in range(electrodes_num):
                trials_dataset[electrode,subjects_through_trials==subjects[i],:] = (trials_dataset[electrode,subjects_through_trials==subjects[i],:])/np.nanstd(trials_dataset[electrode,subjects_through_trials==subjects[i],:])


    clusters = {
            'frontal': ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF8', 'AF3', 'AF4', 'AFz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz'],
            'frontal-central': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz'],
            'central': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz'],
            'parietal-occipital': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'Pz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'O1', 'O2', 'Oz', 'Iz'],
            'temporal': ['TP7', 'TP8','T7', 'T8', 'FT7', 'FT8']
        }
    electrodes_names = electrodes_names
    
    
    sub_electrodes_list = get_frontal_central_indices(electrodes_names, clusters)
    if choose_electrodes:
        trials_dataset = trials_dataset[sub_electrodes_list,:,:]
        electrodes_num = len(sub_electrodes_list)

    
    if cut_beginning and not cut_end:
        trials_dataset = trials_dataset[:,:,20:]        
    elif cut_beginning and cut_end:
        trials_dataset = trials_dataset[:,:,35:45]        
    elif seperate_length:
        nan_mask = np.isnan(trials_dataset)
        non_nan_counts = np.sum(~nan_mask, axis=2)
        lengthes, labels = np.unique(non_nan_counts, return_inverse=True)        
        trials_dataset_paddded  = np.nan*np.empty(trials_dataset.shape)
        for length in lengthes:
            length_trials_indices = np.where(non_nan_counts[0,:]==length)[0]
            trials_dataset_paddded[:,length_trials_indices,:length-25-5]= trials_dataset[:,length_trials_indices,:length-25-5] #trials_dataset_paddded[:,length_trials_indices,:15]= trials_dataset[:,length_trials_indices,length-15:length],trials_dataset[:,length_trials_indices,-4:] ), axis = 2)
        trials_dataset = trials_dataset_paddded


    if concatenate_details_to_data:
        trials_dataset = concatenate_dataset_and_details(trials_dataset, primes, cues, true_labels, subjects_through_trials, electrodes_num)
    
    return trials_dataset, true_labels, cues, primes, subjects_through_trials, trials_sum_new, electrodes_num


   

def concatenate_dataset_and_details(trials_dataset_cutted, primes, cues, true_labels, subjects_through_trials, electrodes_num):
    dataset_and_details = np.concatenate((trials_dataset_cutted, expand_details_vactor(primes, electrodes_num), expand_details_vactor(cues, electrodes_num), expand_details_vactor(true_labels, electrodes_num), expand_details_vactor(subjects_through_trials, electrodes_num)), axis = 2)
    return dataset_and_details

def expand_details_vactor(vector, electrodes_num):
    vector = np.tile(vector[np.newaxis, :], (electrodes_num, 1))[:,:,np.newaxis]
    return vector

 

def dict_dataset_to_uniform_array(dataset,relevant_subjects, electrodes_num, trials_sum, max_trial_length):

    subject_to_subject_id_map = subject_to_subject_id_mapping(relevant_subjects)
    trials_dataset = np.nan*np.empty((electrodes_num, trials_sum,max_trial_length))
    true_labels = np.zeros((trials_sum))
    cues = np.zeros((trials_sum))
    primes = np.zeros((trials_sum))
    subjects_through_trials = np.zeros((trials_sum))
    experiment_types = dataset.keys()
    start_idx,end_idx = 0,0
    for experiment_type in experiment_types:
        labels = dataset[experiment_type].keys()
        for label in labels:
            label_id = label_to_int_id(label[2])
            cue_id = label_to_int_id(label[1])
            prime_id = label_to_int_id(label[0])
            subjects = dataset[experiment_type][label].keys()
            for subject in subjects:
                subject_id = subject_to_subject_id_map.subject_to_subject_id(int(subject))
                subject_exp = dataset[experiment_type][label][subject]
                _,number_of_trials, trial_length = np.shape(subject_exp)              
                rng = np.random.default_rng()
                rng.shuffle(subject_exp, axis=1)  
                           
                trials_dataset[:,start_idx:end_idx +number_of_trials,0:trial_length] = subject_exp
 
                true_labels[start_idx:end_idx +number_of_trials] = label_id*np.ones(number_of_trials)
                cues[start_idx:end_idx +number_of_trials] = cue_id*np.ones(number_of_trials)
                primes[start_idx:end_idx +number_of_trials] = prime_id*np.ones(number_of_trials)
                subjects_through_trials[start_idx:end_idx +number_of_trials] = subject_id*np.ones(number_of_trials)
                start_idx += number_of_trials
                end_idx+= number_of_trials 
    
    return trials_dataset, true_labels, cues, primes, subjects_through_trials

def get_frontal_central_indices(electrodes_names, clusters):
    frontal_names = clusters['frontal']
    central_names = clusters['central']

    # Combine frontal and central names
    target_names = set(frontal_names + central_names)

    # Get indices in electrodes_names
    indices = [i for i, name in enumerate(electrodes_names) if name in target_names]

    return indices


class subject_to_subject_id_mapping():
    def __init__(self,relevant_subjects):
        experiment_types = relevant_subjects.keys()
        subjects = []

        for experiment_type in experiment_types:
            subjects.append(relevant_subjects[experiment_type])
        subjects = np.concatenate(subjects,axis=0)
        subjects_number = len(subjects)
        subjects_ids_map = dict(zip(subjects,np.arange(0,len(subjects))))
        self.subjects_ids_map = subjects_ids_map
        self.subjects_number = subjects_number
            
    def subject_to_subject_id(self, subject):
        subjects_ids_map = self.subjects_ids_map
        subject_id = subjects_ids_map[subject]
        return subject_id


def label_to_int_id(label):
    label_to_int = {'R':1,'L':2, 'N':3, '+':4}
    label_id = label_to_int[label]
    return label_id

def filter_multi_emg_artifacts(trials_dataset, true_labels, cues, primes, subjects_through_trials, window_samples=100):
    """
    Filter trials with artifacts specifically for the MULTI_EMG group.
    
    Rejection Criteria:
    - >100 µV absolute difference between samples within 100 ms intervals
    - Absolute amplitude beyond the ±100 µV range
    
    Parameters:
    - trials_dataset: numpy array of shape (electrodes, trials, time_steps)
    - sampling_rate: sampling rate in Hz (default 1000 Hz)
    
    Returns:
    - Filtered dataset with artifact trials removed from MULTI_EMG group only
    """
    relevant_subjects = {
        'MULTI_EMG':[506,508,509,510,511,513,514,515,516,517,518,519,520,521,522,523]
    }
    
    subject_mapper = subject_to_subject_id_mapping(relevant_subjects)
    multi_emg_subject_ids = []
    for subject in relevant_subjects['MULTI_EMG']:
        multi_emg_subject_ids.append(subject_mapper.subject_to_subject_id(subject))
    
    multi_emg_indices = np.where(np.isin(subjects_through_trials, multi_emg_subject_ids))[0]
    artifact_indices = []
    
    for trial_idx in multi_emg_indices:
        trial_data = trials_dataset[:, trial_idx, :]
        
        if np.nanmax(np.abs(trial_data)) > 100:
            artifact_indices.append(trial_idx)
            continue
        
        trial_length = np.sum(~np.isnan(trial_data[0, :]))
        artifact_found = False
        
        for electrode in range(trial_data.shape[0]):
            electrode_data = trial_data[electrode, :int(trial_length)]
            electrode_data = electrode_data[~np.isnan(electrode_data)]
            
            if len(electrode_data) < window_samples:
                continue
                
            for start_idx in range(len(electrode_data) - window_samples + 1):
                window_data = electrode_data[start_idx:start_idx + window_samples]
                if np.max(window_data) - np.min(window_data) > 100:
                    artifact_found = True
                    break
            
            if artifact_found:
                break
        
        if artifact_found:
            artifact_indices.append(trial_idx)
    
    artifact_indices = np.array(artifact_indices)
    all_indices = np.arange(trials_dataset.shape[1])
    clean_indices = np.setdiff1d(all_indices, artifact_indices)

    return clean_indices
    


def sliding_window_augmentation(data, labels, window_size=1500, step_size=50, augmentation_factor=6):
    """
    Perform data augmentation using the sliding window technique.
    
    Parameters:
    - data: numpy array of shape (num_samples, num_electrodes, time_steps)
    - window_size: size of the sliding window (in time steps)
    - step_size: how much the window slides at each step
    - augmentation_factor: number of times to augment each sample
    
    Returns:
    - augmented_data: numpy array with augmented data
    """
    num_samples, num_electrodes, time_steps = data.shape
    augmented_data = []
    augmented_labels = []
    
    # Apply sliding window to each sample
    for sample_idx in range(num_samples):
        sample = data[sample_idx, :, :] 
        sample_label = labels[sample_idx] 
        augmented_sample = []
        
        # Generate new samples by sliding the window
        for i in range(augmentation_factor):
            start = np.random.randint(0, time_steps - window_size + 1)
            end = start + window_size
            augmented_sample.append(sample[:, start:end])
            augmented_labels.append(sample_label) 
        augmented_data.append(np.stack(augmented_sample, axis=1))
    # Concatenate augmented samples
    augmented_data = np.concatenate(augmented_data, axis=1)
    augmented_labels = np.array(augmented_labels)  
    return augmented_data, augmented_labels