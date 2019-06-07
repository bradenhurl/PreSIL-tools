import os
import sys
endTrain = 44000
endIndex = 51073
valIndex = 44001

#Put indices here with errors in ground planes (or other errors)
emptyIndices = [20821, 42591]

# Set to 1 for full validation set
valSkip = 5#Only print every x indices

print("Indices to skip: ", emptyIndices)

data_dir = os.path.expanduser('~') + '/GTAData/object/training/label_2'
files = os.listdir(data_dir)
num_files = len(files)
file_idx = 0
for file in os.listdir(data_dir):
    filepath = data_dir + '/' + file
    if os.stat(filepath).st_size == 0:
        idx = int(os.path.splitext(file)[0])
        emptyIndices.append(idx)

    sys.stdout.write("\rWorking on idx: {} / {}".format(
            file_idx + 1, num_files))
    sys.stdout.flush()
    file_idx = file_idx + 1

print(emptyIndices)
def printIdx(idx, f):
    if idx not in emptyIndices:
        idxStr = "%06d\n" % idx
        f.write(idxStr)

tv_file = os.path.expanduser('~') + '/GTAData/object/trainval.txt'
with open(tv_file, 'w+') as f:
    for idx in range(0,endIndex + 1):
        printIdx(idx, f)

t_file = os.path.expanduser('~') + '/GTAData/object/train.txt'
with open(t_file, 'w+') as f:
    for idx in range(0,endTrain):
        printIdx(idx, f)

val_full_file = os.path.expanduser('~') + '/GTAData/object/val_full.txt'
with open(val_full_file, 'w+') as f:
    for idx in range(valIndex,endIndex + 1):
        printIdx(idx, f)

val_file = os.path.expanduser('~') + '/GTAData/object/val.txt'
with open(val_file, 'w+') as f:
    for idx in range(valIndex,endIndex + 1, valSkip):
        printIdx(idx, f)
