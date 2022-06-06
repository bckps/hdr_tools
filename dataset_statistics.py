import os, json
import numpy as np

store_data = {
    'CumulativeDegree-66ns': 0,
    'CumulativeDegree-99%-time': 0,
    'CumulativeDegree-99dot9%-time': 0,
    'min-depth': 0,
    'avr-depth': 0,
    'max-depth': 0,
}

cumu_99_num = 0
cumu_999_num = 0
min_depth = 100
avr_depth = 0
max_depth = 0

min_depth_p = 100
avr_depth_p = 0
max_depth_p = 0

min_depth_nz = 100
avr_depth_nz = 0
max_depth_nz = 0
median_array = np.empty(0)

train_phase_error = 0
train_phase1st_error = 0

if __name__ == '__main__':

    # dataset_name = 'test_rev2'
    # save_folder = os.path.join('datasets',dataset_name)
    # os.makedirs(save_folder, exist_ok=True)
    # npzes_list = ['bathroom-extended-train','bathroom-train','contemporaly-bathroom-extended-train',
    #               'contemporaly-bathroom-train', 'livingroom-extended-train', 'livingroom-train']
    # npzes_list = ['bathroom-extended-train2','bathroom-train2','contemporaly-bathroom-extended-train2',
    #               'contemporaly-bathroom-train2', 'livingroom-extended-train2', 'livingroom-train2']
    # npzes_list = ['bathroom-extended-train','bathroom-train']
    # npzes_list = ['bathroom-train']
    # npzes_list = ['contemporaly-bathroom-extended-train','contemporaly-bathroom-train']
    # npzes_list = ['contemporaly-bathroom-train']
    # npzes_list = ['livingroom-extended-train', 'livingroom-train']
    # npzes_list = ['livingroom-train']
    # npzes_list = ['bedroom-test','contemporaly-bathroom-test','livingroom-test']
    # npzes_list = ['bedroom-test']
    # npzes_list = ['contemporaly-bathroom-test']
    # npzes_list = ['livingroom-test']

    # npzes_list = ['bathroom-small-simu-20images', 'export-contemporaly-bathroom', 'living-room-20images',
    #               'bathroom-filled-wall-train-extended', 'contemporaly-bathroom-train-extended',
    #               'livingroom-train-extended']

    npzes_list = ['v3-data-3rdb/v3-div-cont-val','v3-data-3rdb/v3-contemporary-bathroom-val']  #dataset_name = 'v3-data-cont-val'
    # npzes_list = ['v3-data-3rdb/v3-contemporary-bathroom-train','v3-data-3rdb/v3-div-ex-cont-train']  #dataset_name = 'v3-corner-cont-train'
    # npzes_list = ['v3-data-3rdb/v3-contemporary-bathroom-test','v3-corner-plane-no-del-3rdb']  #dataset_name = 'v3-cont-corner-test'

    assign_number = 0

    for npz_folder in npzes_list:
        for file in os.listdir(os.path.join('npz', npz_folder)):
            for f in os.listdir(os.path.join('npz', npz_folder, file)):
                if 'phase-' in f:
                    print(os.path.join('npz', npz_folder, file, f))
                    with np.load(os.path.join('npz', npz_folder, file,f)) as data:
                        phase = data.get('arr_0')
                # if 'phase1st' in f:
                #     print(os.path.join('npz', npz_folder, file, f))
                #     with np.load(os.path.join('npz', npz_folder, file,f)) as data:
                #         phase1st = data.get('arr_0')
                if 'truth' in f:
                    print(os.path.join('npz', npz_folder, file, f))
                    with np.load(os.path.join('npz', npz_folder, file,f)) as data:
                        truth = data.get('arr_0')
                    median_array = np.insert(median_array, 0, truth.flatten())
                if os.path.splitext(f)[-1] == '.json':
                    # print(assign_number)
                    with open(os.path.join('npz', npz_folder, file, f)) as f:
                        df = json.load(f)
                    cumu_66 = df['CumulativeDegree-66ns']
                    cumu_99_time = df['CumulativeDegree-99%-time']
                    cumu_999_time = df['CumulativeDegree-99dot9%-time']
                    # temp_min = df['min-depth']
                    # temp_mean = df['avr-depth']
                    # temp_max = df['max-depth']
                    # p_min = df['min-depth-p']
                    # p_mean = df['avr-depth-p']
                    # p_max = df['max-depth-p']
                    nz_min = df['min-depth-nz']
                    nz_mean = df['avr-depth-nz']
                    nz_max = df['max-depth-nz']
                    if cumu_99_time < 66.6:
                        cumu_99_num += 1
                    if cumu_999_time < 66.6:
                        cumu_999_num += 1

                    # if temp_min < min_depth:
                    #     min_depth = temp_min
                    # avr_depth += temp_mean
                    # if max_depth < temp_max:
                    #     max_depth = temp_max
                    #
                    # if p_min < min_depth_p:
                    #     min_depth_p = p_min
                    # avr_depth_p += p_mean
                    # if max_depth_p < p_max:
                    #     max_depth_p = p_max

                    if nz_min < min_depth_nz:
                        min_depth_nz = nz_min
                    avr_depth_nz += nz_mean
                    if max_depth_nz < nz_max:
                        max_depth_nz = nz_max


            train_phase_error = np.mean(phase - truth)
            # train_phase1st_error = np.mean(phase1st - truth)
            assign_number += 1

    print('cumu_99_num',cumu_99_num)
    print('cumu_999_num', cumu_999_num)
    # print('min_depth', min_depth)
    # print('avr_depth', avr_depth/assign_number)
    # print('max_depth', max_depth)
    print('phase_error',train_phase_error)
    print('phase1st',train_phase1st_error)
    # print('min_depth', min_depth_p)
    # print('avr_depth', avr_depth_p/assign_number)
    # print('max_depth', max_depth_p)
    print('min_depth', min_depth_nz)
    print('avr_depth', avr_depth_nz/assign_number)
    print('max_depth', max_depth_nz)
    print('median:', median_array.shape)
    print('median:', np.median(median_array))

    # dataset_path = 'datasets/test_rev2'
    # files = []
    # for file in os.listdir(dataset_path):
    #     data_paths = [os.path.join(dataset_path, file, f) for f in os.listdir(os.path.join(dataset_path, file))]
    #     files.append(data_paths)
    # print(len(files))
    # for idx in range(len(files)):
    #     img_paths = files[idx]
    #     for impath in img_paths:
    #
    #         if 'A0' in impath:
    #             with np.load(impath) as data:
    #                 A0_img = data.get('arr_0')
    #         elif 'A1' in impath:
    #             with np.load(impath) as data:
    #                 A1_img = data.get('arr_0')
    #         elif 'A2' in impath:
    #             with np.load(impath) as data:
    #                 A2_img = data.get('arr_0')
    #         elif 'truth' in impath:
    #             with np.load(impath) as data:
    #                 depth_1024 = data.get('arr_0')
    #         elif 'phase' in impath:
    #             with np.load(impath) as data:
    #                 phase_img = data.get('arr_0')
