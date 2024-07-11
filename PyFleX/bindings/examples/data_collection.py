import numpy as np
import time
import multiprocessing as mp

dt = 0.001

# 42: Buoyancy
# 47: DamBreak 5cm
# 48: DamBreak 10cm

def collect_data(info):

    import pyflex

    n_rollout, time_step, n_particles = info['n_rollout'], info['time_step'], info['n_particles']
    positions = np.zeros((n_rollout, time_step, n_particles, 4))
    velocities = np.zeros((n_rollout, time_step, n_particles, 3))
    phases = np.zeros((n_rollout, time_step, n_particles, 1))

    pyflex.init()

    for i in range(n_rollout):

        pyflex.set_scene(0, info['idx'])

        for j in range(time_step):
            positions[i][j] = pyflex.get_positions().reshape(-1, 4)
            velocities[i][j] = pyflex.get_velocities().reshape(-1, 3)
            phases[i][j] = pyflex.get_phases().reshape(-1, 1)

            pyflex.step()

    pyflex.clean()

    return positions, velocities, phases


infos = []
for i in range(5):
    infos.append({'idx': i, 'n_particles': 6683, 'n_rollout': 3, 'time_step': 400})

cores = 5
pool = mp.Pool(processes=cores)
data = pool.map(collect_data, infos)

for i in range(len(data)):
    print(data[i][0].shape, data[i][1].shape, data[i][2].shape)

