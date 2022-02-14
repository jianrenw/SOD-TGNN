from nuscenes import NuScenes
from nuscenes.utils import splits

class nuscenes_split():
    def __init__(self, mode, nusc_trainval, nusc_test):
        self.scene_names_train = splits.train
        self.scene_names_val = splits.val
        self.scene_names_test = splits.test
        self.mode = mode
        self.name_token = {}
        for scene in nusc_trainval.scene:
            self.name_token[scene['name']] = scene['token']
        for scene in nusc_test.scene:
            self.name_token[scene['name']] = scene['token']
        self.nusc_trainval = nusc_trainval
        self.nusc_test = nusc_test
    def get_labeled_train(self):
        return self.scene_names_train[:50]
    def get_unlabeled_train(self):
        if self.mode == 'with_test':
            return self.scene_names_train[50:650] + self.scene_names_val + self.scene_names_test
        elif self.mode == 'without_test':
            return self.scene_names_train[50:650] + self.scene_names_test
    def get_iso(self):  
        return self.scene_names_train[650:]
    def get_test(self):
        return self.scene_names_val
    def get_all_sample_tokens(self, split):
        token_set = {}
        if split == 'labeled_train':
            scene_names = self.get_labeled_train()
        elif split == 'unlabeled_train':
            scene_names = self.get_unlabeled_train()
        elif split == 'iso':
            scene_names = self.get_iso()
        elif split == 'test':
            scene_names = self.get_test()
        for scene_name in scene_names:
            all_sample_tokens = []
            all_timestamps = []
            scene_token = self.name_token[scene_name]
            try:
                my_scene = self.nusc_trainval.get('scene', scene_token)
                nbr_samples = my_scene['nbr_samples']
                sample_token = my_scene['first_sample_token']
                all_sample_tokens.append(sample_token)
                my_sample = self.nusc_trainval.get('sample', my_scene['first_sample_token'])
                all_timestamps.append(my_sample['timestamp'])
                while my_sample['next'] != '':
                    all_sample_tokens.append(my_sample['next'])
                    my_sample = self.nusc_trainval.get('sample', my_sample['next'])
                    all_timestamps.append(my_sample['timestamp'])
                assert nbr_samples == len(all_sample_tokens) and nbr_samples == len(
                    all_timestamps)
                token_set[scene_name] = [all_sample_tokens,all_timestamps]
            except KeyError:
                my_scene = self.nusc_test.get('scene', scene_token)
                nbr_samples = my_scene['nbr_samples']
                sample_token = my_scene['first_sample_token']
                all_sample_tokens.append(sample_token)
                my_sample = self.nusc_test.get('sample', my_scene['first_sample_token'])
                all_timestamps.append(my_sample['timestamp'])
                while my_sample['next'] != '':
                    all_sample_tokens.append(my_sample['next'])
                    my_sample = self.nusc_test.get('sample', my_sample['next'])
                    all_timestamps.append(my_sample['timestamp'])
                assert nbr_samples == len(all_sample_tokens) and nbr_samples == len(
                    all_timestamps)
                token_set[scene_name] = [all_sample_tokens,all_timestamps]
        return token_set


