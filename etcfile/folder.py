import os

def data_folder_path(dataset_path):
    """

    :param dataset_path:
    dataset_path|
                |(scenes)
                |---bathroom|---1/{1st.mat,2nd.mat,3rd.mat}
                |           |---2/{1st.mat,2nd.mat,3rd.mat}
                |
                |---living-room

    :return:data_list
    [{'1st':1st.mat_path,'2nd':1st.mat_path,'3rd':1st.mat_path,'folder_path':ex.'bathroom/1/'},
     {'1st':1st.mat_path,'2nd':1st.mat_path,'3rd':1st.mat_path,'folder_path':ex.'bathroom/2/'},...]


    """
    data_list = []
    scenes = os.listdir(dataset_path)
    for scene in scenes:
        scene_nums = os.listdir(os.path.join(dataset_path,scene)) if os.path.isdir(os.path.join(dataset_path,scene)) else []
        for scene_num in scene_nums:
            if os.path.isfile(os.path.join(dataset_path,scene,scene_num,'1st.mat')):
                data = {'1st':os.path.join(dataset_path,scene,scene_num,'1st.mat'),
                        '2nd':os.path.join(dataset_path,scene,scene_num,'2nd.mat'),
                        '3rd':os.path.join(dataset_path,scene,scene_num,'3rd.mat'),
                        'folder_path':os.path.join(dataset_path,scene,scene_num)}
                data_list.append(data)

    return data_list

if __name__ == '__main__':
    data_list = data_folder_path(r'your path')

    print(data_list[0]['folder_path'].split('\\'))