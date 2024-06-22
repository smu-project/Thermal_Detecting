import torch
import copy
import numpy as np

from PIL import ImageDraw


colors = (( 79, 195, 247),
          (236, 64, 122),
          (126, 87, 194),
          (205, 220, 57),
          (103, 58, 183),
          (255, 160, 0))


def calc_iou(a, b):
    a = a.unsqueeze(0).expand_as(b)

    w = torch.min(a[:, 2], b[:, 2]) - torch.max(a[:, 0], b[:, 0])
    h = torch.min(a[:, 3], b[:, 3]) - torch.max(a[:, 1], b[:, 1])

    a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    intersect = torch.clamp(w, 0.) * torch.clamp(h, 0.)
    union = a + b - intersect

    return intersect / union


def nms(loc, conf, th_iou):
    _, order = conf.sort(descending=True)

    results = []
    while len(order):
        selected = order[0]

        results.append(selected)

        order = order[1:]

        iou = calc_iou(loc[selected], loc[order])
        selected = iou < th_iou
        order = order[selected]

    return torch.LongTensor(results)


def swap(array, left, right):
    array[left], array[right] = array[right], array[left]


def quick_sort(unsorted_arr):
    N = len(unsorted_arr)

    sorted_arr = copy.deepcopy(unsorted_arr)
    sorted_indices = [i for i in range(N)]
    stack = [(0, N-1)]

    while stack:
        L, R = stack[-1]
        stack = stack[:-1]

        if L < R:
            if L + 1 == R:
                if sorted_arr[L] < sorted_arr[R]:
                    swap(sorted_arr, L, R)
                    swap(sorted_indices, L, R)
            else:
                P = (L + R) >> 1
                swap(sorted_arr, R-1, P)
                swap(sorted_indices, R-1, P)
                if sorted_arr[R-1] > sorted_arr[L]:
                    swap(sorted_arr, R-1, L)
                    swap(sorted_indices, R-1, L)
                if sorted_arr[R] > sorted_arr[L]:
                    swap(sorted_arr, R, L)
                    swap(sorted_indices, R, L)
                if sorted_arr[R-1] > sorted_arr[R]:
                    swap(sorted_arr, R-1, R)
                    swap(sorted_indices, R-1, R)

                i = L
                j = R
                pivot = sorted_arr[R]
                while True:
                    i += 1
                    while i < R and pivot < sorted_arr[i]:
                        i += 1

                    j -= 1
                    while j > L and pivot > sorted_arr[j]:
                        j -= 1

                    if j < i:
                        break
                    swap(sorted_arr, i, j)
                    swap(sorted_indices, i, j)

                swap(sorted_arr, R, i)
                swap(sorted_indices, R, i)
                stack.append((L, j))
                stack.append((i+1, R))

    return sorted_arr, sorted_indices


def draw_object_box(img, objects, width=2, file_name=''):
    print("")

    draw = ImageDraw.Draw(img)

    fg = (0, 0, 0)
    bg = (128, 128, 128)

    save_text = []

    save_text.append(f"Image resolution: {img.size[0]}x{img.size[1]}")
    save_text.append("-"*80)
    save_text.append(f"{'idx':>7} {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7} {'score(%)':>10} {'class_number':^15} {'label':^10}")
    save_text.append("-"*80)

    for i, obj in enumerate(objects):
        if not obj:
            continue

        bg = colors[i % len(colors)]

        x1, y1, x2, y2, score, label, class_number = obj

        x1 = int(x1 * img.size[0])
        x2 = int(x2 * img.size[0])
        y1 = int(y1 * img.size[1])
        y2 = int(y2 * img.size[1])

        text = '%s (%.2f)' % (label, score)
        save_text.append(f"{i:>7} {x1:>7} {y1:>7} {x2:>7} {y2:>7} {score*100:>10.2f} {class_number:^15} {label:^10}")

        textsize = draw.textsize(text)

        x0, y0 = (x1, y1-textsize[1]-width*2)
        x1_ = x0 + textsize[0] + 1

        draw.rectangle([x0, y0, x1_, y1], fill=bg, width=width)
        draw.rectangle([x1, y1, x2, y2], outline=bg, width=width)
        draw.text([x0+width, y0+width], text, fill=fg)

    for text in save_text:
        print(text)

    if file_name:
        text_fn = file_name + "_result.txt"
        img_fn = file_name + "_result.png"

        with open(text_fn, "w") as fp:
            for text in save_text:
                print(text, file=fp)

        img.save(img_fn, format="png")

        print("")

        print('Save results (text) {}'.format(text_fn))
        print('Save results (image) {}'.format(img_fn))

    return img


def draw_class(img, label, class_number, file_name=''):
    print('')

    draw = ImageDraw.Draw(img)

    draw.text([0, 0], label)

    save_text = []

    save_text.append("-"*60)
    save_text.append(f'{"class_number":^15} {"label":<10}')
    save_text.append("-"*60)
    save_text.append(f'{class_number:^15} {label:<10}')

    for text in save_text:
        print(text)

    if file_name:
        text_fn = file_name + "_result.txt"
        img_fn = file_name + "_result.png"

        with open(text_fn, "w") as fp:            
            for text in save_text:
                print(text, file=fp)

        img.save(img_fn, format='png')

        print("")

        print('Save results (text) path: {}'.format(text_fn))
        print('Save results (image) path: {}'.format(img_fn))

    return img
