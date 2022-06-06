import os, shutil

if __name__ == '__main__':
    
    # dataset consists of npz list.
    npzes_list = ['v3-data-3rdb/v3-contemporary-bathroom-train','v3-data-3rdb/v3-div-ex-cont-train','v3appendix--cont-bath','v3appendix2-cont-bath']  #dataset_name = 'v3-corner-cont-train',dataset_name = 'v3-4cont-appendix3'

    dataset_name = 'v3-4cont-appendix3'
    save_folder = os.path.join('datasets',dataset_name)
    os.makedirs(save_folder, exist_ok=True)
    assign_number = 0

    for npz_folder in npzes_list:
        for file in os.listdir(os.path.join('npz', npz_folder)):
            for f in os.listdir(os.path.join('npz', npz_folder, file)):
                if os.path.splitext(f)[-1] == '.npz':
                    print(assign_number)
                    print(f)
                    os.makedirs(os.path.join(save_folder, str(assign_number).zfill(5)), exist_ok=True)
                    shutil.copy(os.path.join('npz', npz_folder, file, f), os.path.join(save_folder, str(assign_number).zfill(5), f))
            assign_number += 1

