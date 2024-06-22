default = {
    "network": [],
    "layer": {
        "Input": {},
        "Constant": {},
        "Add": {},
        "Mul": {},
        "Sigmoid": {},
        "Tanh": {},
        "MishResidual": {},
        "Mish": {},
        "Swish": {},
        "Conv2d": {
            "kernel_size": {
                "min": [
                    1,
                    1
                ],
                "max": [
                    7,
                    7
                ],
                "const": [],
                "choices": []
            },
            "stride": {
                "min": [
                    1,
                    1
                ],
                "max": [
                    4,
                    4
                ],
                "const": [],
                "choices": []
            },
            "padding": {
                "min": [
                    0,
                    0
                ],
                "max": [
                    3,
                    3
                ],
                "const": [],
                "choices": []
            },
            "dilation": {
                "min": [
                    1,
                    1
                ],
                "max": [
                    4,
                    4
                ],
                "const": [],
                "choices": []
            },
            "groups": {
                "min": [],
                "max": [],
                "const": 1,
                "choices": []
            }
        },
        "DwConv2d": {
            "kernel_size": {
                "min": [],
                "max": [],
                "const": [
                    3,
                    3
                ],
                "choices": []
            },
            "stride": {
                "min": [
                    1,
                    1
                ],
                "max": [
                    2,
                    2
                ],
                "const": [],
                "choices": []
            },
            "padding": {
                "min": [
                    0,
                    0
                ],
                "max": [
                    1,
                    1
                ],
                "const": [],
                "choices": []
            },
            "dilation": {
                "min": [],
                "max": [],
                "const": [
                    1,
                    1
                ],
                "choices": []
            }
        },
        "MaxPool2d": {
            "kernel_size": {
                "min": [
                    2,
                    2
                ],
                "max": [
                    14,
                    14
                ],
                "const": [],
                "choices": []
            }
        },
        "AvgPool2d": {
            "kernel_size": {
                "min": [
                    2,
                    2
                ],
                "max": [
                    14,
                    14
                ],
                "const": [],
                "choices": []
            }
        },
        "AdaptiveMaxPool2d": {
            "output_size": {
                "min": [],
                "max": [],
                "const": [
                    1,
                    1
                ],
                "choices": []
            }
        },
        "AdaptiveAvgPool2d": {
            "output_size": {
                "min": [],
                "max": [],
                "const": [
                    1,
                    1
                ],
                "choices": []
            }
        },
        "Upsample": {
            "scale_factor": {
                "min": [],
                "max": [],
                "const": [
                    2,
                    2
                ],
                "choices": []
            },
            "mode": {
                "min": [],
                "max": [],
                "const": [],
                "choices": [
                    "zerofill",
                    "nearest"
                ]
            }
        },
        "ByPass": {},
        "Concat": {
            "dim": {
                "min": [],
                "max": [],
                "const": 1,
                "choices": []
            }
        },
        "Flatten": {},
        "Permute": {
            "perms": {
                "min": [],
                "max": [],
                "const": [
                    0,
                    2,
                    3,
                    1
                ],
                "choices": []
            }
        },
        "Reshape": {},
        "Slice": {
            "axes": {
                "min": [],
                "max": [],
                "const": 1,
                "choices": []
            },
            "steps": {
                "min": [],
                "max": [],
                "const": 1,
                "choices": []
            }
        }
    }
}