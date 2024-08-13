import requests
import os
from pathlib import Path
import pickle
from shutil import unpack_archive

urls = dict()
urls['space_shuttle']=['http://www.cs.ucr.edu/~eamonn/discords/TEK16.txt',
                       'http://www.cs.ucr.edu/~eamonn/discords/TEK17.txt',
                       'http://www.cs.ucr.edu/~eamonn/discords/TEK14.txt']

urls['power_demand']=['http://www.cs.ucr.edu/~eamonn/discords/power_data.txt']

for dataname in urls:
    raw_dir = Path('data/', dataname, 'raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in urls[dataname]:
        filename = raw_dir.joinpath(Path(url).name)
        print('Downloading', url)
        resp =requests.get(url)
        filename.write_bytes(resp.content)
        if filename.suffix=='':
            filename.rename(filename.with_suffix('.txt'))
        print('Saving to', filename.with_suffix('.txt'))
        if filename.suffix=='.zip':
            print('Extracting to', filename)
            unpack_archive(str(filename), extract_dir=str(raw_dir))



    for filepath in raw_dir.glob('*.txt'):
        with open(str(filepath)) as f:
            # Label anomaly points as 1 in the dataset
            labeled_data=[]
            for i, line in enumerate(f):
                tokens = [float(token) for token in line.split()]

                if filepath.name == 'TEK16.txt':
                    tokens.append(1.0) if 4270 < i < 4370 else tokens.append(0.0)
                elif filepath.name == 'TEK17.txt':
                    tokens.append(1.0) if 2100 < i < 2145 else tokens.append(0.0)
                elif filepath.name == 'TEK14.txt':
                    tokens.append(1.0) if 1100 < i < 1200 or 1455 < i < 1955 else tokens.append(0.0)
                elif filepath.name == 'power_data.txt':
                    tokens.append(1.0) if 8254 < i < 8998 or 11348 < i < 12143 or 33883 < i < 34601 else tokens.append(0.0)
                labeled_data.append(tokens)


            # Save the labeled dataset as .pkl extension
            labeled_whole_dir = raw_dir.parent.joinpath('labeled', 'whole')
            labeled_whole_dir.mkdir(parents=True, exist_ok=True)
            with open(str(labeled_whole_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                pickle.dump(labeled_data, pkl)

            # Divide the labeled dataset into trainset and testset, then save them
            labeled_train_dir = raw_dir.parent.joinpath('labeled','train')
            labeled_train_dir.mkdir(parents=True,exist_ok=True)
            labeled_test_dir = raw_dir.parent.joinpath('labeled','test')
            labeled_test_dir.mkdir(parents=True,exist_ok=True)

            if filepath.name == 'power_data.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[15287:33432], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[501:15287], pkl)
            elif filepath.name == 'TEK17.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2469:4588], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[1543:2469], pkl)
            elif filepath.name == 'TEK16.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[521:3588], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[3588:4539], pkl)
            elif filepath.name == 'TEK14.txt':
                with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[2089:4098], pkl)
                with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data[97:2089], pkl)

nyc_taxi_raw_path = Path('dataset/nyc_taxi/raw/nyc_taxi.csv')
labeled_data = []
with open(str(nyc_taxi_raw_path),'r') as f:
    for i, line in enumerate(f):
        tokens = [float(token) for token in line.strip().split(',')[1:]]
        tokens.append(1) if 150 < i < 250 or   \
                            5970 < i < 6050 or \
                            8500 < i < 8650 or \
                            8750 < i < 8890 or \
                            10000 < i < 10200 or \
                            14700 < i < 14800 \
                          else tokens.append(0)
        labeled_data.append(tokens)
nyc_taxi_train_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled','train',nyc_taxi_raw_path.name).with_suffix('.pkl')
nyc_taxi_train_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(nyc_taxi_train_path),'wb') as pkl:
    pickle.dump(labeled_data[:13104], pkl)

nyc_taxi_test_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled','test',nyc_taxi_raw_path.name).with_suffix('.pkl')
nyc_taxi_test_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(nyc_taxi_test_path),'wb') as pkl:
    pickle.dump(labeled_data[13104:], pkl)