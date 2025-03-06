
@LOAD_DATASET.register_module()
class NewGSM8KDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=path)
        else:
            datasets = {}
            for split in ['train', 'test']:
                split_path = os.path.join(path, split + '.jsonl')
                split_path = path
                dataset = []
                with open(split_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line.strip())
                        dataset.append(line)
                datasets[split] = Dataset.from_list(dataset)
            dataset = DatasetDict(datasets)
        return dataset
