import numpy as np


def cross_product(u, v):
    return u[0] * v[1] - u[1] * v[0]


def euclid_distance_squared(A, B):
    return np.sum(np.square(A - B))


def oriented_dist_point_to_seg(P, A, B):
    return ((B[1] - A[1]) * P[0] - (B[0] - A[0]) * P[1] + B[0] * A[1] - B[1] * A[0]) / \
           np.sqrt(euclid_distance_squared(A, B))


def closest_point_on_seg(P, A, B):
    vector = (B - A)
    # normalize
    vector /= np.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    # set vector length and the orientation
    vector *= oriented_dist_point_to_seg(P, A, B)
    # perpendicular
    vector = np.array((-vector[1], vector[0]))
    # move point to the line
    X = P + vector
    AB = euclid_distance_squared(A, B)
    AX = euclid_distance_squared(A, X)
    BX = euclid_distance_squared(B, X)
    # move inside the segment if needed
    if AX > BX:
        return B if AX > AB else X
    else:
        return A if BX > AB else X


class LinearSeparation:
    def __init__(self, points1, points2):
        self.points1 = points1
        self.points2 = points2
        self.segment = None
        self.hull1 = None
        self.hull2 = None

    def quickhull_recursive(self, points, hull, A, B, side, outer_points_i):
        # let A be the point on the left side
        if A[0] < B[0]:
            tmp = A
            A = B
            B = tmp

        next_outer_points_i = []

        max_cross = 0
        max_cross_i = 0
        vector_segment = (A[0] - B[0], A[1] - B[1])

        for index in outer_points_i:
            vector_point = (points[index][0] - B[0], points[index][1] - B[1])

            cross = cross_product(vector_segment, vector_point) * side
            # point is outside
            if cross > 0:
                next_outer_points_i.append(index)

            # the point is the farthest from the segment
            if cross > max_cross:
                max_cross = cross
                max_cross_i = index

        # if the point is found, add it to the hull and call recursion for 2 segments
        if max_cross > 0:
            hull.append(points[max_cross_i])
            self.quickhull_recursive(points, hull, A, points[max_cross_i], side, next_outer_points_i)
            self.quickhull_recursive(points, hull, points[max_cross_i], B, side, next_outer_points_i)

    def quickhull(self, points):
        hull = []

        # get maximal and minimal x coordinate
        min_x_i = 0
        max_x_i = 0
        for i in range(len(points)):
            if points[i, 0] < points[min_x_i, 0]:
                min_x_i = i
            if points[i, 0] > points[max_x_i, 0]:
                max_x_i = i

        # add them to the hull
        hull.append(points[min_x_i])
        hull.append(points[max_x_i])

        # remaining points that are not inside the hull
        outer_points_i = range(len(points))

        self.quickhull_recursive(points, hull, points[min_x_i], points[max_x_i], +1, outer_points_i)
        self.quickhull_recursive(points, hull, points[min_x_i], points[max_x_i], -1, outer_points_i)

        np_hull = np.array(hull)
        # sort points to create polygon
        center = np.mean(np_hull, axis=0)
        angles = np.arctan2(np_hull[:, 1] - center[1], np_hull[:, 0] - center[0])
        return np_hull[np.argsort(angles, kind='heapsort')]

    def separate(self):
        self.hull1 = self.quickhull(self.points1)
        self.hull2 = self.quickhull(self.points2)

        min_dist = np.inf
        segment = []
        for h1, h2 in [[self.hull1, self.hull2], [self.hull2, self.hull1]]:
            len2 = len(h2)
            for P in h1:
                side = None
                changed = False
                for i in range(len2):
                    # edge of the hull AB
                    A = h2[i]
                    B = h2[(i + 1) % len2]

                    side_now = cross_product(B - A, B - P) > 0

                    if side is not None and not changed:
                        changed = side != side_now
                    side = side_now

                    closest = closest_point_on_seg(P, A, B)
                    dist = euclid_distance_squared(P, closest)
                    if dist < min_dist:
                        min_dist = dist
                        segment = [P, closest]

                # the point P is inside the hull h2 therefore sets are linearly inseparable
                if not changed:
                    return None

        # implicit definition of separating line
        normal_vector = segment[0] - segment[1]
        a = normal_vector[0]
        b = normal_vector[1]
        midpoint = np.mean(segment, axis=0)
        c = - a * midpoint[0] - b * midpoint[1]
        self.segment = segment
        return {'a': a, 'b': b, 'c': c}


if __name__ == '__main__':
    n1 = 20
    n2 = 30

    points1 = np.random.rand(n1, 2) - 1
    points2 = np.random.rand(n2, 2)

    points1[0] = (0.5, 0.5)

    separation = LinearSeparation(points1, points2)
    separation.separate()
