import math

import numpy as np


def cartesian_to_spherical(x, y, z):
    """
    Calculates spherical coordinates from cartesian coordinates.
    :param x: The x coordinate
    :param y: The y coordinate
    :param z: The z coordinate
    :return: The azimuthal angle, the polar angle and the length.
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi, theta = norm_cartesian_to_spherical(x / r, y / r, z / r)
    return phi, theta, r


def spherical_to_cartesian(phi, theta, r):
    """
    Calculates cartesian coordinates from spherical coordinates.
    :param phi: The azimuthal angle.
    :param theta: The polar angle.
    :param r: The length.
    :return: The x, y, z coordinates.
    """
    x, y, z = norm_spherical_to_cartesian(phi, theta)
    return r * x, r * y, r * z


def norm_cartesian_to_spherical(x, y, z):
    """
        Calculates normalized spherical coordinates from cartesian coordinates.
        :param x: The x coordinate
        :param y: The y coordinate
        :param z: The z coordinate
        :return: The azimuthal angle, the polar angle and the length.
        """
    if x > 0 and y > 0:
        phi = np.arctan(np.fabs(y / x))
    elif x > 0 and y < 0:
        phi = 2 * np.pi - np.arctan(np.fabs(y / x))
    elif x < 0 and y < 0:
        phi = np.pi + np.arctan(np.fabs(y / x))
    elif x < 0 and y > 0:
        phi = np.pi - np.arctan(np.fabs(y / x))
    elif x == 0 and y < 0:
        phi = 3 * np.pi / 2
    elif x == 0 and y > 0:
        phi = np.pi / 2
    else:
        phi = 0

    theta = np.arccos(z)

    return math.degrees(phi), math.degrees(theta)


def norm_spherical_to_cartesian(phi, theta):
    """
    Calculates normalized cartesian coordinates from spherical coordinates.
    :param phi: The azimuthal angle.
    :param theta: The polar angle.
    :return: The x, y, z coordinates.
    """
    theta = math.radians(theta)
    phi = math.radians(phi)
    return math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)
