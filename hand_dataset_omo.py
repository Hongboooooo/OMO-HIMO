import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import h5py

class OMODataset(Dataset):
    def __init__(self, data_type="train", num_objs=2): 
        self.data_type = data_type
        self.num_objs = num_objs
        self.data_path = f"./himo_data/{self.num_objs}o_{self.data_type}.h5"
        print("data_path", self.data_path)
        self.data_names = []
        self.load_data_names()
    
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, index):
        data2return = self.data_names[index]

        with h5py.File(self.data_path, 'r') as f:
            
            single_data = {}
            single_data["seq_name"] = data2return
            single_data["seq_len"] = f[data2return]["seq_len"][()]
            single_data["seq_scale"] = f[data2return]["seq_scale"][()]
            single_data["hands_positions_seq"] = f[data2return]["hands_positions_seq"][:]

            objs_name = []
            for ki in f[data2return]["bps_object_geo_seq"].keys():
                objs_name.append(ki)

            for oi in range(self.num_objs):
                single_data[f"o{oi+1}_name"] = objs_name[oi]
                single_data[f"obj_center_seq_o{oi+1}"] = f[data2return]["obj_center_seq"][objs_name[oi]][:]
                single_data[f"bps_object_geo_seq_o{oi+1}"] = f[data2return]["bps_object_geo_seq"][objs_name[oi]][:]
                single_data[f"rotation_seq_o{oi+1}"] = f[data2return]["object_rotation_seq"][objs_name[oi]][:]
                single_data[f"translation_seq_o{oi+1}"] = f[data2return]["object_translation_seq"][objs_name[oi]][:]

        # for ki in single_data.keys():
        #     print(f"{ki} : {type(single_data[ki])}")

        return single_data

    def load_data_names(self):
        with h5py.File(self.data_path, 'r') as f:
            for ki in f.keys():
                self.data_names.append(ki)

        print(f"length of {self.data_type} data:", self.__len__())
        

# omods = OMODataset(data_type="train", num_objs=2)

# omodl = DataLoader(dataset=omods, batch_size=32, shuffle=True, num_workers=2)

# for i in omodl:
#     print("seq_len:", i["seq_len"])
#     print("seq_name:", i["seq_name"])


        
