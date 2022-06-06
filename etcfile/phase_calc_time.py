import os, shutil, time
import numpy as np

if __name__ == '__main__':
    dataset_name = 'test_rev2'
    # save_folder = os.path.join('datasets',dataset_name)
    # os.makedirs(save_folder, exist_ok=True)
    # npzes_list = ['bathroom-extended-train','bathroom-train','contemporaly-bathroom-extended-train',
    #               'contemporaly-bathroom-train', 'livingroom-extended-train', 'livingroom-train']
    npzes_list = ['bedroom-test','contemporaly-bathroom-test','livingroom-test']
    # npzes_list = ['bathroom-small-simu-20images', 'export-contemporaly-bathroom', 'living-room-20images',
    #               'bathroom-filled-wall-train-extended', 'contemporaly-bathroom-train-extended',
    #               'livingroom-train-extended']
    assign_number = 0

    calc_time_list = []
    A0 = None
    A1 = None
    A2 = None


    for npz_folder in npzes_list:
        for file in os.listdir(os.path.join('npz', npz_folder)):
            for f in os.listdir(os.path.join('npz', npz_folder, file)):
                if 'A0' in f:
                    print(os.path.join('npz', npz_folder, file,f))
                    with np.load(os.path.join('npz', npz_folder, file,f)) as data:
                        A0 = data.get('arr_0')
                if 'A1' in f:
                    print(os.path.join('npz', npz_folder, file, f))
                    with np.load(os.path.join('npz', npz_folder, file,f)) as data:
                        A1 = data.get('arr_0')
                if 'A2' in f:
                    print(os.path.join('npz', npz_folder, file, f))
                    with np.load(os.path.join('npz', npz_folder, file,f)) as data:
                        A2 = data.get('arr_0')
            assign_number += 1
            if A0.any() and A1.any() and A2.any():
                c = 299792458
                eps = 1e-5
                z_coef = c * 22.2 * (10 ** -9) / 2.0

                start = time.time()
                depth_from_phase = np.where(A0 < A2, z_coef * ((A2 - A0) / (A1 + A2 - 2 * A0 + eps) + 1),
                                            z_coef * (A1 - A2) / (A0 + A1 - 2 * A2 + eps))
                calc_time = time.time() - start
                calc_time_list.append(calc_time)
                A0 = None
                A1 = None
                A2 = None
    print(assign_number)
    print(np.mean(calc_time_list))

                    # print(f)
                    # os.makedirs(os.path.join(save_folder, str(assign_number).zfill(5)), exist_ok=True)
                    # shutil.copy(os.path.join('npz', npz_folder, file, f), os.path.join(save_folder, str(assign_number).zfill(5), f))


