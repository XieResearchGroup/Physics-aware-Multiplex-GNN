import os
from tqdm import tqdm

DATASET_PATH = "/home/mjustyna/RNA-GNN/data/RNA-PDB-clean/"

def main():
    ignored_ids_file = "all_ignored.ids"
    with open(ignored_ids_file, "r") as f:
        ignored_ids = f.readlines()
        ignored_ids = [id.strip() for id in ignored_ids]
    
    dirs = os.listdir(DATASET_PATH)
    for d in dirs:
        print("Removing from", d)
        for id in tqdm(ignored_ids):
            path = os.path.join(DATASET_PATH, d, id+".pkl")
            if os.path.exists(path):
                os.remove(path)
    pass


if __name__ == "__main__":
    main()