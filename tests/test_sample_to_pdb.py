import pytest
from torch_geometric.loader import DataLoader
from datasets import RNAPDBDataset
from utils import SampleToPDB

class TestSampleToPDB:
    data_path = "data/RNA-PDB/"
    out_path = "tests/test_output/"

    @pytest.mark.skip(reason="Test not implemented")
    def test_to_pdb(self):
        # Test the to_pdb method
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='all')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to_pdb(data, self.out_path, name)
            break
        # Add assertions to verify the output

    def test_write_xyz_all(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='all')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to_xyz(data, self.out_path, name, post_fix='_all')
            break

    def test_write_xyz_backbone(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='backbone')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to_xyz(data, self.out_path, name, post_fix='_bb')
            break

    def test_write_xyz_coarse(self):
        val_ds = RNAPDBDataset(self.data_path, name='val-raw-pkl', mode='coarse-grain')
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        sample = SampleToPDB()

        for data, name in val_loader:
            sample.to_xyz(data, self.out_path, name, post_fix='_cgr')
            break
