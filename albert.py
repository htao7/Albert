import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

SMALLEST_RADIUS = 20
COMPLETION = 0.55


def ExportSize(core_size, count):
    file = open('size.txt', 'w')
    file.write("Index \t radius \n")
    for i in range(0, count):
        file.write("%s \t %s \n" % (str(i + 1), str(core_size[i])))
    file.close()


img0 = cv2.imread('UV exposed.bmp')
# img0 = cv2.imread('Picture1.png')
im = np.copy(img0)
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7, 7), 0)
kernel = np.ones((2, 2), np.uint8)
wells_mask = np.zeros(img.shape, np.uint8)

wells = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200, \
                         param1=160, param2=40, minRadius=50, maxRadius=150)
wells = np.uint16(np.around(wells))

count = 0
core_size = []
for i in wells[0, :]:
    well_center = (i[0], i[1])
    well_radius = int(i[2] * 1.1)
    well_i = np.copy(img[well_center[1] - well_radius:well_center[1] + well_radius, \
                     well_center[0] - well_radius:well_center[0] + well_radius])

    edges = cv2.Canny(well_i, 30, 50)
    edges = cv2.dilate(edges, kernel)

    # cv2.imshow('well',edges)
    # cv2.waitKey(0)

    r_score = []
    previous_similarity = 0
    previous_nearcenter = False
    previous_center = (0, 0)
    for r in range(SMALLEST_RADIUS, well_radius, 1):
        template = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
        template = cv2.circle(template, (r, r), r, 255, 1)
        res = cv2.matchTemplate(edges, template, cv2.TM_CCORR)
        (_, score, _, loc) = cv2.minMaxLoc(res)
        res_identical = cv2.matchTemplate(template, template, cv2.TM_CCORR)
        (_, score_identical, _, _) = cv2.minMaxLoc(res_identical)
        similarity = score / score_identical
        nearcenter = ((well_radius - 2 * r - 1) < loc[0] < well_radius) \
                     and ((well_radius - 2 * r - 1) < loc[1] < well_radius)
        r_score.extend([similarity])
        if previous_similarity > COMPLETION and similarity < previous_similarity and previous_nearcenter is True:
            cv2.circle(im, previous_center, r - 1, (0, 0, 255), 2)
            count += 1
            cv2.putText(im, str(count), previous_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            core_size.extend([r])
            break
        previous_similarity = similarity
        previous_nearcenter = nearcenter
        previous_center = (well_center[0] - well_radius + loc[0] + r, \
                           well_center[1] - well_radius + loc[1] + r)

    # Find the first local maxima

    # plt.subplot(221),plt.imshow(well_i,'gray')
    # plt.subplot(222),plt.imshow(edges, 'gray')
    # plt.subplot(212),plt.plot(range(SMALLEST_RADIUS,well_radius),r_score)
    # plt.show()

    # cv2.circle(im,(well_center[0] - well_radius + core_loc[0] + core_radius, well_center[1] - well_radius + core_loc[1] + core_radius), \
    #            core_radius,(0,0,255),2)

    # cv2.circle(wells_mask, well_center, well_radius, 255, -1)

# res = cv2.bitwise_and(img, img, mask=wells_mask)
# edges_all = cv2.Canny(res, 30, 50)


# cv2.imshow('labelled',im)
# cv2.imshow('edge',edges_all)
# cv2.waitKey()
# cv2.destroyAllWindows()

cv2.imwrite('labelled.png', im)
ExportSize(core_size, count)
# cv2.imwrite('edges.png',edges_all)
