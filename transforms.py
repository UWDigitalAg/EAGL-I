"""
Functions to convert between image and real world coordinates

Version 0.4
Author: Chen-Yi Liu, Michael Beck
Date: June 24, 2019
University of Winnipeg
"""

import math
import numpy as np

# specifications of GoPro Hero 7 camera and WidowX gimbal
GIMBAL_ARM_LEN = 9.0  # cm
LENS_LENGTH = 0.0  # cm
LENS_OFFSET = 1.0  # cm for mounted GoPro
LINEAR_FOV_H = 86.7  # degrees  GoPro official: 86.7
LINEAR_FOV_V = 71.0  # degrees  GoPro official: 71.0
WIDE_FOV_H = 122.6  # degrees  GoPro official: 122.6
WIDE_FOV_V = 94.4  # degrees  GoPro official: 94.4
PIXELS_H = 4000  # GoPro resolution
PIXELS_V = 3000  # GoPro resolution


# Rotations and translation matrices
def matrix_elements(angle):
    angle = math.radians(angle)
    return math.cos(angle), math.sin(angle)


def rotate_x(pos, angle):
    c, s = matrix_elements(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]])
    new_pos = np.matmul(rotation_matrix, pos)
    return new_pos


def rotate_y(pos, angle):
    c, s = matrix_elements(angle)
    rotation_matrix = np.array([[c, 0, s],
                                [0, 1, 0],
                                [-s, 0, c]])
    new_pos = np.matmul(rotation_matrix, pos)
    return new_pos


def rotate_z(pos, angle):
    c, s = matrix_elements(angle)
    rotation_matrix = np.array([[c, -s, 0],
                                [s, c, 0],
                                [0, 0, 1]])
    new_pos = np.matmul(rotation_matrix, pos)
    return new_pos


def translate(pos, displacement):
    return np.asarray(pos + displacement)


# coordinate transform functions
def camera_displacement(gantry_pos, gimbal_pan, gimbal_tilt):
    """
    calculate the position of the camera in world frame for a given
    gantry position and gimbal pan and tilt angles
    gantry_pos: numpy array of size (3,) in centimeters
    gimbal_pan: pan angle in degrees
    gimbal_tilt: tilt angle in degrees
    """

    pos = np.array([GIMBAL_ARM_LEN, LENS_OFFSET, -LENS_LENGTH])
    pos = rotate_y(pos, gimbal_tilt)
    pos = rotate_z(pos, gimbal_pan)
    return translate(pos, gantry_pos)


def get_camera_offset(pan, tilt):
    """
    calculates the movement of the camera relative to the gantry when panning
    and tilting
    pan: pan angle in degrees
    tilt: tilt angle in degrees
    """
    pos = np.array([GIMBAL_ARM_LEN, LENS_OFFSET, -LENS_LENGTH])
    pos = rotate_y(pos, tilt)
    pos = rotate_z(pos, pan)
    return pos[0], pos[1]


def to_camera_frame(obj_pos, gantry_pos, gimbal_pan, gimbal_tilt):
    """
    convert the position of the object from world frame to camera frame for a
    given gantry position and gimbal pan&tilt angles.
    obj_pos: object position in world frame. numpy array of
                            size (3,), in cm
    gantry_pos: numpy array of size (3,) in centimeters
    gimbal_pan: pan angle in degrees
    gimbal_tilt: tilt angle in degrees
    object
    """
    camera_pos = camera_displacement(gantry_pos, gimbal_pan, gimbal_tilt)
    obj_pos = translate(obj_pos, -camera_pos)
    obj_pos = rotate_z(obj_pos, -gimbal_pan)
    obj_pos = rotate_y(obj_pos, -gimbal_tilt)
    # our configuration has the camera mounted clock-wise at 90 degrees
    obj_pos = rotate_z(obj_pos, 90.0)
    return obj_pos


def bearing(obj_pos):
    """
    Return bearing of an object:
    https://en.wikipedia.org/wiki/Bearing_(navigation). Used to determine
    an objects center in x-direction (more to the left or right in the image)
    """
    angle = math.atan2(obj_pos[0], -obj_pos[2])
    return math.degrees(angle)


def elevation(obj_pos):
    """
    Return elevation angle of an object:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system. Used to determine
    an objects center in y-direction (more to the top or bottom in the image)
    """
    angle = math.atan2(obj_pos[1], -obj_pos[2])
    return math.degrees(angle)


def subtended_angle(size, distance):
    """
    Return the subtended angle angle of a sphere with certain size and distance:
    https://en.wikipedia.org/wiki/Subtended_angle. Used to determine the size of
    the bounding box located at a certain position in the image.
    """
    angle = math.fabs(size) / math.fabs(distance)
    return math.degrees(angle)


def to_view_angles(obj_pos, obj_radius=0.0):
    """
    in camera frame, the spherical object occupies the field of view with
    an arc proportional to the object radius.
    obj_pos: object position in camera frame. numpy array of shape (3,), in cm
    obj_radius: radius of the spherical object, in cm
    """
    distance = np.linalg.norm(obj_pos)
    obj_subtended = subtended_angle(obj_radius, distance)
    obj_bear = bearing(obj_pos)
    obj_elev = elevation(obj_pos)
    return obj_bear, obj_elev, obj_subtended


def angle_to_pixel(angle, fov, num_pixels):
    """
    angle: object angle
    fov: field of view of the camera
    num_pixels: number of pixels along the dimension that fov was measured
    """
    if angle > 90.0 or angle < -90.0:
        raise ValueError

    x = math.tan(math.radians(angle))
    limit = math.tan(math.radians(fov / 2))
    return int(x / limit * num_pixels / 2 + num_pixels / 2)


# Logical operations on bounding boxes
def intersect(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Calculates the intersection of two rectangles.
    First return value indicates whether there is an overlap between the two.
    Second return value is a numpy array of shape (4,) giving the two diagonal
    corners' coordinates of the intersection. First corner is the one closest to
    the origin and the second one is the one furthest away from the origin.
    """
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x4 = min(x1 + w1, x2 + w2)
    y4 = min(y1 + h1, y2 + h2)
    if x3 < x4 and y3 < y4:
        return True, np.array([x3, y3, x4, y4])
    return False, None


def contain(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    check if the first rectangle fully contains the second one
    """
    return x1 <= x2 and y1 <= y2 and x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1


# functions to be called by users of this module
def to_image_coordinates(obj_pos, gantry_pos, gimbal_pan, gimbal_tilt):
    """
    Calculates the position of the object in camera image.
    obj_pos: object position in world frame. numpy array of size (3,), in cm
    gantry_pos: gantry position in world frame. numpy array of size (3,), in cm
    Gimbal angle is defined as the direction in which a light ray coming from
    the origin enters straight into the camera lens.
    gimbal_pan: gimbal azimuthal angle in degrees
    gimbal_tilt: gimbal polar angle in degrees
    Returns coordinates as a pair of pixel values. First value increases from
    left to right of the image. The second value increases from top to bottom of
    the image. Returns None if the object is behind the camera.
    """
    obj_pos = to_camera_frame(obj_pos, gantry_pos, gimbal_pan, gimbal_tilt)
    obj_bear, obj_elev, _ = to_view_angles(obj_pos, 0.0)
    if obj_bear > 90.0 or obj_bear < -90.0 or \
            obj_elev > 90.0 or obj_elev < -90.0:
        return
    x = angle_to_pixel(obj_bear, LINEAR_FOV_H, PIXELS_H)
    y = angle_to_pixel(-obj_elev, LINEAR_FOV_V, PIXELS_V)
    return np.array([x, y])


def within_view(gantry_pos, gimbal_pan, gimbal_tilt, obj_pos, obj_radius=0.0):
    """
    Check if the spherical object is entirely within the field of view of the
    camera.
    gantry_pos: gantry position in world frame. numpy array of size (3,), in cm
    gimbal_pan: gimbal azimuthal angle in degrees
    gimbal_tilt: gimbal polar angle in degrees
    obj_pos: object position in world frame. numpy array of size (3,), in cm
    obj_radius: object radius in cm
    Return a boolean
    """
    obj_pos = to_camera_frame(obj_pos, gantry_pos, gimbal_pan, gimbal_tilt)
    obj_bear, obj_elev, obj_subtended = to_view_angles(obj_pos, obj_radius)
    top = obj_elev + obj_subtended
    bottom = obj_elev - obj_subtended
    left = obj_bear - obj_subtended
    right = obj_bear + obj_subtended
    return contain(-LINEAR_FOV_H / 2, -LINEAR_FOV_V / 2, LINEAR_FOV_H, LINEAR_FOV_V,
                   left, bottom, right - left, top - bottom)


def bounding_box(gantry_pos, gimbal_pan, gimbal_tilt, obj_pos, obj_radius=0.0):
    """
    Gives a bounding box of the spherical object in the image.
    gantry_pos: gantry position in world frame. numpy array of size (3,), in cm
    gimbal_pan: gimbal azimuthal angle in degrees
    gimbal_tilt: gimbal polar angle in degrees
    obj_pos: object position in world frame. numpy array of size (3,), in cm
    obj_radius: object radius in cm
    Return
    First value indicates whether any portion of the bounding box falls within
    the field of view of the camera.
    Second value is the coordinates of the top-left and bottom-right corners of
    the bounding box. numpy array containing (top-left x, top-left y,
    bottom-right x, bottom-right y)
    """
    coordinates = to_image_coordinates(obj_pos, gantry_pos,
                                       gimbal_pan, gimbal_tilt)
    if coordinates is None:
        return False, None
    obj_pos = to_camera_frame(obj_pos, gantry_pos, gimbal_pan, gimbal_tilt)
    _, _, obj_subtended = to_view_angles(obj_pos, obj_radius)
    half_length = int(obj_subtended / LINEAR_FOV_V * PIXELS_V)
    overlap, coordinates = intersect(0, 0, PIXELS_H, PIXELS_V,
                                     coordinates[0] - half_length,
                                     coordinates[1] - half_length,
                                     2 * half_length, 2 * half_length)
    if not overlap:
        return False, None
    return True, coordinates


def to_rotated_image_coordinates(coordinates):
    """
    When the camera is upside down, use this function to convert pixel
    coordinates to the rotated ones.
    """
    x1 = PIXELS_H - coordinates[0]
    y1 = PIXELS_V - coordinates[1]
    if len(coordinates) == 2:
        return np.array([x1, y1])
    elif len(coordinates) == 4:
        x2 = PIXELS_H - coordinates[2]
        y2 = PIXELS_V - coordinates[3]
        return np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])


def occluded(bounding_boxes, occluded_area=1.0):
    """
    Check if any of the bounding boxes is occluded by another box by more than a
    certain percentage.
    Input
    bounding_boxes: a list containing the bounding boxes. Each item contains
    four coordinate values, namely x1, y1, x2, y2. The bounding boxes are
    ordered from near to far from the camera lens.
    occluded_area: a float between 0.0 and 1.0, indicating how much a bounding
    box needs to be occluded before it is eliminated from the list. If the value
    is 0.5, the bounding box needs to be occluded by at least 50% for it to be
    eliminated.
    Return
    a list of boolean indicating whether each box is occluded
    """
    occluded = [False] * len(bounding_boxes)
    for i, box1 in enumerate(bounding_boxes):
        for j in range(i + 1, len(bounding_boxes)):
            box2 = bounding_boxes[j]
            x1, y1 = box1[0], box1[1]
            w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
            x2, y2 = box2[0], box2[1]
            w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

            overlap, intersection = intersect(x1, y1, w1, h1, x2, y2, w2, h2)
            if overlap:
                width = intersection[2] - intersection[0]
                height = intersection[3] - intersection[1]
                if (width * height) / (w2 * h2) >= occluded_area:
                    occluded[j] = True

    return occluded


def determine_camera_distance(obj_radius):
    """
    the distance of the object from the camera when the object fills the field
    of view.
    """
    fov = min(LINEAR_FOV_H, LINEAR_FOV_V)
    return obj_radius / math.tan(math.radians(fov / 2))


def determine_gantry_position(obj_pos, azimuth, polar, distance):
    """
    Calculates where the gantry needs to be in order to capture the object from
    certain azimuthal and polar angle, and at certain distance.
    obj_pos: object position in world frame. numpy array of size (3,), in cm
    azimuth: the azimuthal angle, in degrees, from which to capture an image of
    the object
    polar: polar angle, in degrees, from which to capture an image of the object
    distance: distance between the object and the camera, in cm
    Return numpy array of shape (3,), in cm. gantry position in world frame
    """
    r = distance
    s_t = math.sin(math.radians(polar))
    c_t = math.cos(math.radians(polar))
    s_p = math.sin(math.radians(azimuth))
    c_p = math.cos(math.radians(azimuth))
    camera_pos = np.array([r * s_t * c_p, r * s_t * s_p, r * c_t])
    camera_pos = translate(camera_pos, obj_pos)
    origin = np.array([0, 0, 0])
    gimbal_arm = camera_displacement(origin, azimuth, polar)
    gantry_pos = translate(camera_pos, -gimbal_arm)
    return gantry_pos


def determine_camera_orientation(obj_pos, gantry_pos):
    """
    Calculates the orientation of the camera for a given object position and
    gantry position.
    obj_pos: object position in world frame. numpy array of size (3,), in cm
    gantry_pos: gantry position in world frame. numpy array of size (3,), in cm
    Return a pair of values indicating the pan and tilt angles of the gimbal
    in degrees.
    """
    obj_pos = translate(obj_pos, -gantry_pos)
    gimbal_pos = -obj_pos
    r = np.linalg.norm(gimbal_pos)
    theta = math.acos(gimbal_pos[2] / r)
    phi = math.atan2(gimbal_pos[1], gimbal_pos[0])
    pan = math.degrees(phi)
    if pan < 0.0:
        pan += 360.0
    tilt = math.degrees(theta)
    x_offset, y_offset = get_camera_offset(pan, tilt)
    return pan, tilt, x_offset, y_offset
