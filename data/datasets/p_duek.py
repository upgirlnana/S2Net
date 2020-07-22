import os.path as  osp
import glob
import re
class P_DUKE(object):
    """
    P_DUKE


    """
    # root = "D://dataset//Partial_iLIDS//"
    # root="/home/rxn/Dataset/partial/P-DukeMTMC-reid/test/"
    root='/home/rxn/dataset/partial/P-DukeMTMC-reid'
    train_dir = osp.join('/home/rxn/dataset/partial/P-DukeMTMC-reid/train/', 'bounding_box_train')
    # '/home/rxn/dataset/partial/P-DukeMTMC-reid/train/bounding_box_train'
    query_dir = osp.join('/home/rxn/dataset/partial/P-DukeMTMC-reid/test', 'occluded_images')
    gallery_dir = osp.join('/home/rxn/dataset/partial/P-DukeMTMC-reid/test', 'whole_images')


    def __init__(self, **kwargs):
        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_gallery_pids + num_query_pids
        num_total_imgs =num_query_imgs + num_gallery_imgs

        print("=> P_DUKE dataset loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        # salience directories
        # self.salience_dir = osp.join(self.root, 'salience')
        self.salience_train_dir = osp.join(self.root, 'occluded_parsing')
        self.salience_query_dir = osp.join(self.root, 'occluded_parsing')
        self.salience_gallery_dir = osp.join(self.root, 'whole_parsing')

        # semantic parsing directories
        # self.parsing_dir = osp.join(self.root, 'parsing')
        self.parsing_train_dir = osp.join("/home/rxn/dataset/partial/P-DukeMTMC-reid/train/", 'train_parsing')
        # self.parsing_train_dir = osp.join(self.root,'occluded_parsing')
        self.parsing_query_dir = osp.join('/home/rxn/dataset/partial/P-DukeMTMC-reid/test', 'occluded_parsing')
        self.parsing_gallery_dir = osp.join('/home/rxn/dataset/partial/P-DukeMTMC-reid/test', 'whole_parsing')

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        pattern = re.compile(r'([-\d]+)_([\d]+)')


        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # import pdb
            # pdb.set_trace()
            # p=pid
            # camid = pid
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 1000 # pid == 0 means background
            # assert 1 <= camid <= 119
            # print(img_path.split('/')[-2])

            fii = img_path
            # a = (img_path.split('//')[-2]).split('_')[0]
            # import pdb
            # pdb.set_trace()
            if ((img_path.split('/')[-2])).split('_')[0] == 'occluded':
                camid = 1
                # print('True')
            else:
                camid = 0
            # if (img_path.split('/')[-2])=='Gallery':
            #     camid=1
            #     print('True')
            # else:
            #     camid=0
            # camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

