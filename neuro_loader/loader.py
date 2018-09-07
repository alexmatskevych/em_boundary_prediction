from inferno.io.core import Zip, Concatenate
from neurofire.datasets.cremi.loaders import RawVolume,SegmentationVolume
from inferno.io.transform import Compose
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform
from neurofire.transform.affinities import affinity_config_to_transform
from inferno.utils.io_utils import yaml2dict
from inferno.io.transform.generic import AsTorchBatch

from torch.utils.data.dataloader import DataLoader


# The dataloader for one Neuro data block
class NeuroLoader(Zip):

    def __init__(self, name, volume_config, slicing_config, master_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert 'segmentation' in volume_config

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))
        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolume(name=name, **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name,
                                                      **segmentation_volume_kwargs)

        # Initialize zip
        super().__init__(self.raw_volume,
                         self.segmentation_volume,
                        sync=True)

        # Set master config (for transforms)
        self.master_config = {} if master_config is None else master_config
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):

        if self.master_config.get('perform_trafos', True):

            transforms = Compose(RandomFlip3D(), RandomRotate())

            # Elastic transforms can be skipped by setting elastic_transform to false in the
            # yaml config file.
            if self.master_config.get('elastic_transform'):
                elastic_transform_config = self.master_config.get('elastic_transform')
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

        else:
            transforms = Compose()

        if self.affinity_config is not None and self.master_config.get('use_affinities', True):
            # we apply the affinity target calculation only to the segmentation (1)
            transforms.add(affinity_config_to_transform(apply_to=[1],
                                                            **self.affinity_config))

        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   master_config=master_config)


class NeuroLoaders(Concatenate):
    def __init__(self, names,
                 volume_config,
                 slicing_config,
                 master_config=None):
        # Make datasets and concatenate
        datasets = [NeuroLoader(name=name,
                                      volume_config=volume_config,
                                      slicing_config=slicing_config,
                                      master_config=master_config)
                    for name in names]
        # Concatenate
        super().__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('dataset_names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   slicing_config=slicing_config, master_config=master_config)


def get_neuro_loaders(config):
    """
    Gets Neuro loaders given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    config = yaml2dict(config)
    datasets = NeuroLoaders.from_config(config)
    loader = DataLoader(datasets, **config.get('loader_config'))
    return loader
