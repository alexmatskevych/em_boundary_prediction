import torch
import h5py
import numpy as np

from inferno.utils.io_utils import yaml2dict

from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform
from neurofire.transform.affinities import affinity_config_to_transform
from inferno.io.transform import Compose
from inferno.trainers.basic import Trainer
from inferno.io.transform.generic import Cast, Normalize

def test_elastic_trafo():

    def get_transforms():

        if master_config.get('perform_trafos', True):

            transforms = Compose(RandomFlip3D(), RandomRotate())

            # Elastic transforms can be skipped by setting elastic_transform to false in the
            # yaml config file.
            if master_config.get('elastic_transform'):
                elastic_transform_config = master_config.get('elastic_transform')
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

        else:
            transforms = Compose()

        if affinity_config is not None and master_config.get('use_affinities', True):
            # we apply the affinity target calculation only to the segmentation (1)
            transforms.add(affinity_config_to_transform(**affinity_config))

        return transforms

    gt_block_path="/g/kreshuk/data/fib25_blocks/gt/gt_block1.h5"
    gt_h5_path="data"

    config = yaml2dict("./configs/affinities_with_trafo/data_config.yml")
    names = config.get('dataset_names')
    volume_config = config.get('volume_config')
    slicing_config = config.get('slicing_config')
    master_config = config.get('master_config')

    # Get kwargs for raw volume
    raw_volume_kwargs = dict(volume_config.get('raw'))
    raw_volume_kwargs.update(slicing_config)

    # Get kwargs for segmentation volume
    segmentation_volume_kwargs = dict(volume_config.get('segmentation'))
    segmentation_volume_kwargs.update(slicing_config)
    affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)

    # Set master config (for transforms)
    master_config = {} if master_config is None else master_config
    # Get transforms
    transforms = get_transforms()

    with h5py.File(gt_block_path, "r") as f:
        block=np.array(f[gt_h5_path])

    block_trafo=transforms(block)

    print("done")


def test_boundary_preds():
    path_to_tr="/g/kreshuk/matskevych/boundary_map_prediction/project_folder_old/Weights/"
    path_to_h5="/g/kreshuk/data/fib25_blocks/raw/raw_block8.h5"
    path_to_raw="/g/kreshuk/matskevych/boundary_map_prediction/project_folder_old/Weights/raw_best.h5"
    path_to_pred="/g/kreshuk/matskevych/boundary_map_prediction/project_folder_old/Weights/pred_best.h5"

    tr=Trainer()
    tr.load(path_to_tr,best=True)
    tr.cuda()
    tr.eval_mode()

    file=h5py.File(path_to_h5, "r")
    transforms = Compose(Normalize(mean=None, std=None))


    block=np.array(file["data"][:160,:160,:160]).astype("float32")

    block_transformed=transforms(block)

    block_torch=torch.from_numpy(block_transformed).unsqueeze(0).unsqueeze(0).cuda()

    pred=tr.apply_model(block_torch)

    with h5py.File(path_to_raw) as f:
        f.create_dataset("data",data=block_transformed,compression="gzip")

    with h5py.File(path_to_pred) as f:
        f.create_dataset("data",data=pred.cpu().detach().numpy(),compression="gzip")



if __name__ == '__main__':
    test_boundary_preds()