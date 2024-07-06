# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

# Extract number from a file and get a plot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os

def draw_plot(data, save_dir, multiple=False):
    plt.figure()

    # plt.plot(data, color='magenta', marker='o',mfc='pink' ) #plot the data
    if multiple:
        for i in range(len(data)):
            plt.plot(data[i], label=f'layer-{i}')
        plt.legend()
    else:
        plt.plot(data) #plot the data
    # plt.xticks(range(0,len(data)+1, 1)) #set the tick frequency on x-axis

    plt.ylabel('loss') #set the label for y axis
    plt.xlabel('epoch') #set the label for x-axis

    plt.savefig(save_dir)

def extract_losses_from_file(loss_name,file,add_prefix=True):
    given_file = open(file, 'r')
    lines = given_file.readlines()

    loss_epoch = []
    for line in lines:
        if add_prefix:
            key_name = f"{loss_name}: "
        else:
            key_name = loss_name
        if key_name in line:
            # extract
            begin = line.find(key_name) + len(key_name)
            end = line.find(' (', begin)
            num_str = line[begin:end]
            if 'Time' in key_name:
                num_str = num_str[:-1]
            num = float(num_str)
            loss_epoch.append(num)
    given_file.close()

    return loss_epoch

def main():
    # Load file
    file_log = "valid_full_dataset.txt"
    file_name = f"/X/{file_log}"
    loss_list = ['loss_pose_perjoint', 'loss_ce', 'class_error', 'loss_pose_perprojection', 
        'loss_pose_perprojection_2d', 
        'cardinality_error', 'gradnorm', 'loss_pose_perbone', 'losses_all', 'Time']

    save_dir = f'./process/save/{file_log}'
    os.makedirs(save_dir, exist_ok=True)

    for loss_name in loss_list:
        if loss_name == 'gradnorm':
            add_prefix = False
        else:
            add_prefix = True
        loss_epoch = extract_losses_from_file(loss_name, file_name, add_prefix=add_prefix)
        # draw plot and save

        draw_plot(loss_epoch, f"{save_dir}/loss-{loss_name}.png")
        print('finish loss:', loss_name)

    # combine layer loss
    layer_num = 2
    loss_list = []
    for ly in range(layer_num):
        layer_loss_name = f'loss_pose_perjoint_l{ly}'
        loss_ly = extract_losses_from_file(layer_loss_name, file_name, add_prefix=True)
        loss_list.append(loss_ly)
    # draw all data in a plot
    draw_plot(loss_list, f"{save_dir}/loss-pose_perjoint-layers.png", multiple=True)



if __name__ == "__main__":
    main()