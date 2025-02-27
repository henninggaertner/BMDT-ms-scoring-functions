# read an mgf file line by line, if the line begins with a float followed by a space and then "0.0", skip the line.
# all other lines should be written to a new file prepended with the string "new_".

import os
from tqdm import tqdm
file_list = ["data/CTR03_BA46_INSOLUBLE_01.mgf", "data/CTR08_BA46_INSOLUBLE_01.mgf", "data/CTR45_BA46_INSOLUBLE_01.mgf"]

for file in file_list:
    with open(file, 'r') as f:
        new_file = "new_" + os.path.basename(file)
        with open(new_file, 'w') as new_f:
            for line in tqdm(f, desc="Processing " + file, unit="lines"):
                if line.split(" ")[-1] == "0.0\n":
                    continue
                new_f.write(line)


