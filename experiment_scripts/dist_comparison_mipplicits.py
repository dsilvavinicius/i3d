import torch
from i3d.meshing import gen_mc_coordinate_grid
from i3d.util import from_pth


def compute_delta(N=100000, delta=0.5):
    
    m1 = from_pth("results/mip_arm_coarse/best.pth").eval().cuda()
    m2 = from_pth("results/mip_arm_fine/best.pth").eval().cuda()

    # print(m1)
    # print(m2)


    coords = torch.tensor([]).cuda()
    M = 0

    with torch.no_grad():
        while M < N:
            # coords = gen_mc_coordinate_grid(N, 2.0 / (N - 1), t=None)[..., :3].cuda()
            new_coords = ((torch.rand((N, 3)) * 1.8) - 0.9).cuda()
            # coords *= 0.8
            id = torch.abs(m2(new_coords)['model_out']) < delta
            id = id.squeeze()
            if not coords.shape[0]:
                coords = new_coords[id, ...].clone()
            else:
                coords = torch.cat((coords, new_coords[id, ...]), dim=0)

            M = coords.shape[0]

        coords = coords[N-1, ...]

        y1 = m1(coords)["model_out"]
        y2 = m2(coords)["model_out"]

        sup = torch.max(torch.abs(y1 - y2))
        # print(sup.item())
        return sup


deltas = [0.1,0.2,0.3,0.4,0.5]
sample_sizes=[1000, 5000, 10000, 50000, 100000, 200000]

for n in sample_sizes:
    print("for N = ", n)
    for delta in deltas:
        iters = 1000

        norms = torch.zeros(size=(iters, 1))
        for i in range(iters):
            norms[i] = compute_delta(n, delta)

        print("mean of the sup norms for delta ", delta, " = ", norms.mean())