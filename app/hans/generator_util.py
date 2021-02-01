# import cv2
# import numpy as np


# def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):

#     if dx > 0:
#         image = np.pad(
#             image, ((0, 0), (dx, 0), (0, 0)), constant_values=255)
#     else:
#         image = image[:, -dx:, :]

#     if (new_w + dx) < net_w:
#         image = np.pad(
#             image, ((0, 0), (0, net_w - (new_w+dx)), (0, 0)), constant_values=255)

#     if dy > 0:
#         image = np.pad(
#             image, ((dy, 0), (0, 0), (0, 0)), constant_values=255)
#     else:
#         image = image[-dy:, :, :]

#     if (new_h + dy) < net_h:
#         image = np.pad(
#             image, ((0, net_h - (new_h+dy)), (0, 0), (0, 0)), constant_values=255)

#     return image[:net_h, :net_w, :]


# def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
#     # determine scale factors
#     dhue = np.random.uniform(-hue, hue)
#     dsat = _rand_scale(saturation)
#     dexp = _rand_scale(exposure)

#     # convert RGB space to HSV space
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

#     # change satuation and exposure
#     image[:, :, 1] *= dsat
#     image[:, :, 2] *= dexp

#     # change hue
#     image[:, :, 0] += dhue
#     image[:, :, 0] -= (image[:, :, 0] > 180)*180
#     image[:, :, 0] += (image[:, :, 0] < 0) * 180

#     # convert back to RGB from HSV
#     return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


# def _rand_scale(scale):
#     scale = np.random.uniform(1, scale)
#     return scale if (np.random.randint(2) == 0) else 1./scale


# def random_flip(image, flip):
#     if flip == 1:
#         return np.flip(image, (0, 1))
#     return image
