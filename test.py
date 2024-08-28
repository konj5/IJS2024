from tqdm import tqdm
import time

for i in tqdm(range(20)):
    for j in tqdm(range(20), leave=False):
        time.sleep(0.1)