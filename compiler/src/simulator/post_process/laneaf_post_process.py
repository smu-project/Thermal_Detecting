import torch
import numpy as np

from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment


def tensor2image(tensor, mean, std):
    mean = mean[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)
    std = std[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)

    image = 255.0*(std*tensor[0].cpu().float().numpy() + mean) # (nc, H, W)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0)) # (C, H, W) to (H, W, C)
    image = image[:, :, ::-1] # RGB to BGR
    return image.astype(np.uint8) # (H, W, C)


def decodeAFs(BW, VAF, HAF, fg_thresh=128, err_thresh=5):
    output = np.zeros_like(BW, dtype=np.uint8) # initialize output array
    lane_end_pts = [] # keep track of latest lane points
    next_lane_id = 1 # next available lane ID

    # start decoding from last row to first
    for row in range(BW.shape[0]-1, -1, -1):
        cols = np.where(BW[row, :] > fg_thresh)[0] # get fg cols
        clusters = [[]]
        if cols.size > 0:
            prev_col = cols[0]

        # parse horizontally
        for col in cols:
            if col - prev_col > err_thresh: # if too far away from last point
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0: # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0: # found lane center, process VAF
                clusters[-1].append(col)
                prev_col = col
            elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0: # found lane end, spawn new lane
                clusters.append([])
                clusters[-1].append(col)
                prev_col = col
                continue
            elif HAF[row, prev_col] < 0 and HAF[row, col] < 0: # keep moving to the right
                clusters[-1].append(col)
                prev_col = col
                continue

        # parse vertically
        # assign existing lanes
        assigned = [False for _ in clusters]
        C = np.Inf*np.ones((len(lane_end_pts), len(clusters)), dtype=np.float64)
        for r, pts in enumerate(lane_end_pts): # for each end point in an active lane
            for c, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                # mean of current cluster
                cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                # get vafs from lane end points
                vafs = np.array([VAF[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                vafs = vafs / np.linalg.norm(vafs, axis=1, keepdims=True)
                # get predicted cluster center by adding vafs
                pred_points = pts + vafs*np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                # get error between prediceted cluster center and actual cluster center
                error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                C[r, c] = error
        # assign clusters to lane (in acsending order of error)
        row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
        for r, c in zip(row_ind, col_ind):
            if C[r, c] >= err_thresh:
                break
            if assigned[c]:
                continue
            assigned[c] = True
            # update best lane match with current pixel
            output[row, clusters[c]] = r+1
            lane_end_pts[r] = np.stack((np.array(clusters[c], dtype=np.float32), row*np.ones_like(clusters[c])), axis=1)
        # initialize unassigned clusters to new lanes
        for c, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            if not assigned[c]:
                output[row, cluster] = next_lane_id
                lane_end_pts.append(np.stack((np.array(cluster, dtype=np.float32), row*np.ones_like(cluster)), axis=1))
                next_lane_id += 1

    return output


def coord_op_to_ip(x, y, scale):
    # (208*scale, 72*scale) --> (208*scale, 72*scale+14=590) --> (1664, 590) --> (1640, 590)
    if x is not None:
        x = int(scale*x)
        x = x*1640./1664.
    if y is not None:
        y = int(scale*y+14)
    return x, y


def get_lanes_culane(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(589, 240, -10)
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip(x_op, y_op, samp_factor)
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 20:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)

    # get x-coordinates from fitted spline
    lanes = []
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    lane += [_x, _y]
            lanes.append(lane)

    return lanes


def match_multi_class(pred, target):
    pred_ids = np.unique(pred[pred > 0]) # find unique pred ids
    target_ids = np.unique(target[target > 0]) # find unique target ids
    pred_out = np.zeros_like(pred) # initialize output array

    # return input array if no lane points in prediction/target
    if pred_ids.size == 0:
        return pred
    if target_ids.size == 0:
        return pred

    assigned = [False for _ in range(pred_ids.size)] # keep track of which ids have been asssigned

    # create cost matrix for matching predicted with target lanes
    C = np.zeros((target_ids.size, pred_ids.size))
    for i, t_id in enumerate(target_ids):
        for j, p_id in enumerate(pred_ids):
            C[i, j] = -np.sum(target[pred == p_id] == t_id)

    # optimal linear assignment (Hungarian)
    row_ind, col_ind = linear_sum_assignment(C)
    for r, c in zip(row_ind, col_ind):
        pred_out[pred == pred_ids[c]] = target_ids[r]
        assigned[c] = True

    # get next available ID to assign
    if target_ids.size > 0:
        max_target_id = np.amax(target_ids)
    else:
        max_target_id = 0
    next_id = max_target_id + 1 # next available class id
    # assign IDs to unassigned fg pixels
    for i, p_id in enumerate(pred_ids):
        if assigned[i]:
            pass
        else:
            pred_out[pred == p_id] = next_id
            next_id += 1
    assert np.unique(pred[pred > 0]).size == np.unique(pred_out[pred_out > 0]).size, "Number of output lanes altered!"

    return pred_out


def laneaf_post_process(BW, VAF, HAF, input_seg):
    output_scale = 0.25
    samp_factor = 2. / output_scale

    mask_out = tensor2image(torch.sigmoid(BW).repeat(1, 3, 1, 1).detach(),
                            np.array([0.0 for _ in range(3)], dtype='float32'),
                            np.array([1.0 for _ in range(3)], dtype='float32'))

    vaf_out = np.transpose(VAF[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
    haf_out = np.transpose(HAF[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

    # decode AFs to get lane instances
    seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)

    if torch.any(torch.isnan(input_seg)):
        # if labels are not available, skip this step
        pass
    else:
        # if test set labels are available
        # re-assign lane IDs to match with ground truth
        seg_out = match_multi_class(seg_out.astype(np.int64),
                                    input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))

    # get results in output structure
    return get_lanes_culane(seg_out, samp_factor)

