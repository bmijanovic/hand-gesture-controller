import os 


def rename_files(directory, labels=False):
    index = 0
    images = os.listdir(directory)
    images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))
    for image_file in images:
        command, number = image_file.split('_')
        print(index, number)
        if labels:
            os.rename(f'{directory}/{image_file}', f'{directory}/{command}_{index}.jpg.csv')
        else:
            os.rename(f'{directory}/{image_file}', f'{directory}/{command}_{index}.jpg')
        index += 1


rename_files('./data/images/up')
rename_files('./data/labels/up', labels=True)

rename_files('./data/images/down')
rename_files('./data/labels/down', labels=True)

rename_files('./data/images/left')
rename_files('./data/labels/left',  labels=True)

rename_files('./data/images/right')
rename_files('./data/labels/right', labels=True)

rename_files('./data/images/open')
rename_files('./data/labels/open', labels=True)



