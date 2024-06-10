import csv, json
import warnings
import numpy as np

ball_labels = ["Cue ball(white Ball)", "8-ball(black ball)", "Solid ball", "Stripe ball"]
ball_info = [["cue ball", 1], ["8-ball", 1], ["solid ball", 7], ["stripe ball", 7]]
def check_ball_cnt(ball_id, ass_id):
    check_ball_cnt.ballCounts[ball_id] += 1
    cnt = check_ball_cnt.ballCounts[ball_id]
    if cnt > ball_info[ball_id][1]:
        warnings.warn(f"Too many {ball_info[ball_id][0]}s. id={ass_id}")
        return -1
    return cnt
check_ball_cnt.ballCounts = [0, 0, 0, 0]

def calc_coordinates(box_info):
    x = box_info["left"] + box_info["width"] / 2.0
    y = box_info["top"] - box_info["height"] / 2.0
    return x, y

ball_coordinates = []

with open("results/Batch_5229409_batch_results.csv", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[16] == "AssignmentStatus": continue
        if row[16] == "Rejected": continue
        assert row[16] == "Approved", "AssignmentStatus is invalid."

        answer = json.loads(row[28])[0]
        boundingBoxes = answer["annotatedResult"]["boundingBoxes"]
        
        ball_coordinates.append([row[27], np.zeros((16, 2))])
        for box in boundingBoxes:
            idx = ball_labels.index(box["label"])
            cnt = check_ball_cnt(idx, row[14])
            if cnt < 0:
                continue

            x, y = calc_coordinates(box)

            if idx < 2: coo_idx = idx
            else: coo_idx = (idx-2) * 7 + cnt + 1
            ball_coordinates[-1][1][coo_idx] = [x, y]
        check_ball_cnt.ballCounts = [0, 0, 0, 0]


rack_dict = {}
averaged_ball_coordinates = np.empty((0, 16, 2))
is_high = []
for rack_info in ball_coordinates:
    img_name = rack_info[0]
    if img_name not in rack_dict:
        rack_dict[img_name] = [len(is_high), 1] # register [index, count]
        
        y = 0
        if img_name.endswith("_high.png"): y = 1
        else:
            assert img_name.endswith("_low.png"), "image name is invalid."
        
        averaged_ball_coordinates = \
            np.append(averaged_ball_coordinates, [rack_info[1]], axis=0)
        is_high.append(y)
    
    else:
        idx, cnt = rack_dict[img_name]
        rack_dict[img_name] = [idx, cnt + 1]
        averaged_ball_coordinates[idx] += rack_info[1]
        averaged_ball_coordinates[idx] /= cnt + 1
        pass

print(averaged_ball_coordinates)