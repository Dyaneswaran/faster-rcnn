def IoU(a_bbox, b_bbox):

    (xa1, ya1, xa2, ya2) = a_bbox
    (xb1, yb1, xb2, yb2) = b_bbox

    ### Intersection ###

    x1 = max(xa1, xb1)
    y1 = max(ya1, yb1)
    x2 = min(xa2, xb2)
    y2 = min(ya2, yb2)

    intersection = max(0, (x2 - x1 + 1)) * max(0, y2 - y1 + 1)

    ### Union ###

    area1 = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)
    area2 = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)
    union = area1 + area2 - intersection
    
    return intersection/union 