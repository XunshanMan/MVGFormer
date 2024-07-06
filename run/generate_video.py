# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

import cv2

data_source_dir = "/XXX/"

frame_jump = 1
max_frame_num = 319
# max_frame_num = 110
max_batch_num = 8
max_layer_num = 4

frame_rate = 5

frame_list = range(0+frame_jump,max_frame_num,frame_jump)
view_list = range(0,5)

# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = None

for frame_id in frame_list:  # time sequence
    for batch_id in range(0,max_batch_num): # time sequence
        im_list_timestamp = []
        for layer_id in range(0,max_layer_num):
            view_im_list = []
            for view_id in view_list:
                im_name = f"/{frame_id}/layer-{layer_id}/batch_{batch_id}/v{view_id}.png"
                im_name = data_source_dir + im_name
                im = cv2.imread(im_name)

                # Large pic
                view_im_list.append(im)

            # add 3d: notice there are 8 batch
            # im_name_3d = f"/{frame_id}/joints_l{layer_id}_filtered.png"
            # im_name_3d = data_source_dir + im_name_3d
            # im_3d = load_im_3d(im_name_3d, batch_id, max_batch_num)
            # view_im_list.append(im_3d)

            view_im = cv2.hconcat(view_im_list)
            im_list_timestamp.append(view_im)
        # generate large im using all the im in the list
        im_large = cv2.vconcat(im_list_timestamp)

        # add to video
        if out is None:
            video_im_size = (im_large.shape[1], im_large.shape[0])
            print("video size:", video_im_size)
            out = cv2.VideoWriter(f'output_r{frame_rate}_max{max_frame_num}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, video_im_size)
        out.write(im_large)

        # video.add(im_large)
        # print('save image to test.png')
        # cv2.imwrite("test.png", im_large)
    print('finish frame:', frame_id)

# save video
# video.save('./1.mp4')
out.release()
print('done')