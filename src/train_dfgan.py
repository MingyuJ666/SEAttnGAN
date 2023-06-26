import os
import sys

current_cwd = os.getcwd()
src_path = '/'.join(current_cwd.split('/')[:-1])
sys.path.append(src_path)
from train import train
from utils import plot_losses

g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = train()

path = '/Public/FYP_temp/fyp23_hanyu_zhang/loss'

filenames = [path + 'loss1.csv', path + 'loss2.csv', path + 'loss3.csv']

for i in range(len(loss_values)):
    filename = filenames[i]
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Loss'])
        writer.writerow([loss])
