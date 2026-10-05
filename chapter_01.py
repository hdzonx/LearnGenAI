import sys
print(sys.prefix != sys.base_prefix)  # True = venv ativo   
import torch
import numpy as np



heighs = [189, 170, 189, 163, 183, 171, 185,
168, 173, 183, 173, 173, 175, 178,
183, 193, 178, 173, 174, 183, 183,
180, 168, 180, 170, 178, 182, 180,
183, 178, 182, 188, 175, 179, 183,
193, 182, 183, 177, 185, 188, 188,
182, 185, 191, 183]


heighs_tensor = torch.tensor(heighs, dtype=torch.float64)

tensor_1 = torch.zeros(3,3)
print(tensor_1)

tensor_2 = torch.ones(1,3,5)
print(tensor_2)

height_thomas_jeferson = heighs_tensor[2]
print("\n Thomas Jeferson height = ", height_thomas_jeferson)


nparr = np.array(range(10))
pt_tensor = torch.tensor(nparr, dtype=torch.int)
print(pt_tensor)

height_Donald_Trump = heighs_tensor[-2]
print("\n Donald Trump height = ", height_Donald_Trump)
