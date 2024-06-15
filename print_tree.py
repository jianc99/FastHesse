import torch
# torch.set_printoptions(profile="full")
from torch.distributed import _functional_collectives as funcol

funcol.all_reduce()

m68 = torch.load("68m-70b-acc.pt")
b13 = torch.load("1.3b-70b-acc.pt")
b7 = torch.load("7b-70b-acc.pt")
# vec = torch.load("acceptance-rate-vector.pt")
# vec1 = torch.load("btree_acc_1.3b.pt")
# tree = torch.load("demo_tree.pt")
# tree_mask :torch.Tensor = vec["mask"]
# tree_mask = (tree_mask == 0).type(torch.float16)

# tree_mask.masked_fill_(tree_mask > 0, torch.finfo(torch.float16).min)
print(m68)
print(b13)
print(b7)
# print(vec1)