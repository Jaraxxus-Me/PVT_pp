from .otb import OTBDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .uav10fps import UAV10Dataset
from .uavdark import UAVDARKDataset
from .uavdt import UAVDTDataset
from .dtb import DTB70Dataset
from .uav20l import UAV20Dataset
from .uav123 import UAV123Dataset
from .visdrone import VISDRONEDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'DTB70' in name:
            dataset = DTB70Dataset(**kwargs)
        elif 'UAV10' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'VISDRONE' in name:
            dataset = VISDRONEDataset(**kwargs)
        elif 'UAVDT' in name:
            dataset = UAVDTDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAVDARK' in name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'UAV123' in name:
            dataset = UAV123Dataset(**kwargs)
        elif 'UAVDARK' in name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

