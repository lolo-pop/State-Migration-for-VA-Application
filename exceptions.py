from __future__ import print_function, unicode_literals


class KubeException(Exception):
    """ Generic exception when something goes wrong """


class PackageSignatureException(KubeException):
    """ Exception raised when package signature validation goes wrong"""


class ProbeTimeout(KubeException):
    pass


class CountExceeded(KubeException):
    pass