
def metrics_select_online():

    n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

    if (n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
        print('Input does not meet requirements.')
        exit()

    data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2}: '))

    if (data_type not in ['A', 'B']):
        print('Input does not meet requirements.')
        exit()

    data_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2'}
    model_folder_name_dict = data_folder_name_dict

    model_type = str(input('Please input the type of trained model to use {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2}: '))

    if (model_type not in ['A', 'B', 'C']):
        print('Input does not meet requirements.')
        exit()

    return n_days_lookahead, data_type, data_folder_name_dict, model_type, model_folder_name_dict


def metrics_select_offline():
    n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

    if (n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
        print('Input does not meet requirements.')
        exit()

    data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Unbalanced}:  '))
    if (data_type not in ['A', 'B', 'C', 'D']):
        print('Input does not meet requirements.')
        exit()

    data_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'offline_dataset'}
    model_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'offline_model'}

    model_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Unbalanced}: '))

    if (model_type not in ['A', 'B', 'C', 'D']):
        print('Input does not meet requirements.')
        exit()

    return n_days_lookahead, data_type, data_folder_name_dict, model_type, model_folder_name_dict


def train_select_online():
    n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

    if (n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
        print('Input does not meet requirements.')
        exit()

    data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2}: '))
    model_type = data_type

    if (data_type not in ['A', 'B', 'C']):
        print('Input does not meet requirements.')
        exit()

    data_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2'}
    model_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2'}

    return n_days_lookahead, data_type, data_folder_name_dict, model_type, model_folder_name_dict


def train_select_offline():
    n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

    if (n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
        print('Input does not meet requirements.')
        exit()

    data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Unbalanced}:  '))
    model_type = data_type

    if (data_type not in ['A', 'B', 'C', 'D']):
        print('Input does not meet requirements.')
        exit()

    data_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'offline_dataset'}
    model_folder_name_dict = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'offline_model'}

    return n_days_lookahead, data_type, data_folder_name_dict, model_type, model_folder_name_dict
