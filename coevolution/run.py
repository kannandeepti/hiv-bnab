from simulation import Simulation
import warnings
import sys

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    founder_distance = int(sys.argv[1])
    rs = int(sys.argv[2])
    sim = Simulation(founder_distance=founder_distance, seed=rs)
    sim.run()
