import os, json

dataset_path = 'npz/bathroom-small-simu-20images'
jsonlist = []

for num in os.listdir(dataset_path):
    for file in os.listdir(os.path.join(dataset_path,num)):
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.json':
            jsonlist.append(os.path.join(dataset_path,num,file))

            with open(os.path.join(dataset_path,num,file)) as f:
                data = json.load(f)
            if data['CumulativeDegree-66ns'] < 0.999999:
                print(file)
                print(data['CumulativeDegree-66ns'])
                print(data['CumulativeDegree-center-99%-time'])
                print(data['max-depth'])
                print('')
                break

print(len(jsonlist))




