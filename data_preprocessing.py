# %%
import h5py
import numpy as np
import pandas as pd
import os
import json

max_smiles_len = 150
original_csv = 'data/full.csv'


def csv_to_ohe_csv(smiles_df, data_signature, charset, max_smiles_len=max_smiles_len, dataset_type='train'):

    df = smiles_df
    
    # Save one-hot encoded CSVs for train and test
    num_chars = len(charset)
    total_rows = len(df) * max_smiles_len

    # Preallocate a large numpy array
    ohe_array = np.zeros((total_rows, num_chars), dtype=np.float32)

    # Build the index for characters in charset
    char_to_index = {char: i for i, char in enumerate(charset)}

    # Fill the array
    row_idx = 0
    for smiles in df['Smiles']:
        padded_smiles = smiles + ' ' * (max_smiles_len - len(smiles))
        for char in padded_smiles:
            if char in charset:
                ohe_array[row_idx, char_to_index[char]] = 1.0
            row_idx += 1
            if row_idx >= total_rows:  # Safety check to prevent out-of-bounds access
                break
        if row_idx >= total_rows:  # Break the outer loop if limit is reached
            break

    # Convert the numpy array to DataFrame and save
    ohe_df = pd.DataFrame(ohe_array, columns=range(num_chars))
    ohe_csvfile = f'data/ohe_data_{dataset_type}_{data_signature}.csv'
    ohe_df.to_csv(ohe_csvfile, index=False)
    print(f"Saved one-hot encoded CSV to {ohe_csvfile}")


def ohe_csvs_to_h5(data_signature):
    ohe_csvfiles = [f'data/charset_{data_signature}.csv', 
                    f'data/ohe_data_train_{data_signature}.csv', 
                    f'data/ohe_data_test_{data_signature}.csv']

    metadata_file = f'data/metadata_{data_signature}.json'
    # Load metadata for reshaping
    with open(metadata_file, 'r') as meta_file:
        metadata = json.load(meta_file)

    h5file = os.path.join(f'data/processed_{data_signature}.h5')
    with h5py.File(h5file, 'w') as f:
        for csv in ohe_csvfiles:
            if 'charset' in csv:
                key = 'charset'
            if 'data_train' in csv:
                key = 'data_train'
            if 'data_test' in csv:
                key = 'data_test'
            print(f"Processing key: {key}")

            # Load CSV
            df = pd.read_csv(csv, header=None if key == "charset" else 0)

            if key == "charset":
                # Encode charset as fixed-size byte strings
                data = df.iloc[0].astype(str).apply(lambda x: x.encode('utf-8')).to_numpy(dtype="S1")
            else:
                # Load data as float32 for other datasets
                data = df.to_numpy(dtype=np.float32)
                original_shape = tuple(metadata[key]['original_shape'])
                data = data.reshape(original_shape)

            # Save dataset
            f.create_dataset(key, data=data)
            print(f"Saved {key} to {h5file}, shape: {data.shape}")


def original_csv_to_ohe_h5(original_csv, max_smiles_len=max_smiles_len, charset=None):
    df = pd.read_csv(original_csv)

    # save a copy of full.csv without records that have smiles longer than max_smiles_len
    df = df[df['Smiles'].str.len() < max_smiles_len]
    
    if charset is None:
        charset = sorted(set(''.join([' '] + list(df['Smiles']))))

    data_len = len(df)
    print(f'original data length: {data_len}')

    train_len = int(len(df)*0.8)
    print(f'train length: {train_len}')

    test_len = len(df) - train_len
    print(f'test length: {test_len}')

    charset_len = len(charset)
    print(f'charset length: {len(charset)}')

    data_signature = f'{data_len}_{max_smiles_len}_{charset_len}'
    print(f'data signature: {data_signature}')

    # Save charset
    pd.DataFrame([charset]).to_csv(f'data/charset_{data_signature}.csv', index=False, header=False)

    # Save full csv
    df.to_csv(f'data/full_{max_smiles_len}_len.csv', index=False)

    # Shuffle Data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # make metadata file for the new csv
    metadata = {}
    metadata['charset'] = {'original_shape': [charset_len, 1]}
    metadata['data_train'] = {'original_shape': [train_len, max_smiles_len, charset_len]}
    metadata['data_test'] = {'original_shape': [test_len, max_smiles_len, charset_len]}
    with open(f'data/metadata_{data_signature}.json', 'w') as f:
        json.dump(metadata, f)

    # make data_train.csv
    train_df = df.iloc[:train_len]
#    train_df.to_csv(f'data/data_train_{data_signature}.csv', index=False, header=True)

    # make data_test.csv
    test_df = df.iloc[train_len:]
#    test_df.to_csv(f'data/data_test_{data_signature}.csv', index=False, header=True)

    # Save one-hot encoded CSVs for train and test
    csv_to_ohe_csv(train_df, data_signature, charset, max_smiles_len, dataset_type='train')
    csv_to_ohe_csv(test_df, data_signature, charset, max_smiles_len, dataset_type='test')

    # make h5 file
    ohe_csvs_to_h5(data_signature)

    # close file that was open for writing
    f.close()
    

    # return file path
    return f'data/processed_{data_signature}.h5'



def main(original_csv=original_csv):
    original_csv_to_ohe_h5(original_csv, max_smiles_len=max_smiles_len)
    
if __name__ == '__main__':
    main()


# %%


# %%



