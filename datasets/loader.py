from datasets.grid import GridDataset

def get_dataset(data_path, args, rate=1):
    if args.dataset=='GRID':
        return GridDataset(data_path, args, rate)
    return None