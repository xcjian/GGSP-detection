from utils import *
# This function shows the current progress.

# count the number of total tasks
dataset = 'straight_move' # Note: if use 'seismic', it's better to set the inspect instances to instead of [0, 1, 2].
vary_snr = True
config = get_config(dataset, vary_snr)
repeat_time = config['repeat_time']
res_path = config['res_path']
method_names = ['MHT-GGSP', 'MHT-GGSP-oracle']
n_T = len(res_path)
n_methods = len(method_names)

n_total_task = repeat_time * n_T * n_methods

# count the number of completed tasks
n_completed = 0
for res_path_ in res_path:
    try:
        num_files = sum(1 for f in os.listdir(res_path_) if os.path.isfile(os.path.join(res_path_, f)))
        n_completed = n_completed + num_files
    except:
        pass

# print the progress as percentage
percent = n_completed / n_total_task * 100
print(f"Total tasks: {n_total_task} | Completed: {n_completed} | Progress: {percent:.1f}%")
