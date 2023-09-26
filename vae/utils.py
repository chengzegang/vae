from typing import Any
import torch
import torch.library as lib
from enum import Enum


def dtypes(name: str) -> torch.dtype:
    match name:
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case "int32":
            return torch.int32
        case "int64":
            return torch.int64
        case "int16":
            return torch.int16
        case "int8":
            return torch.int8
        case "uint8":
            return torch.uint8
        case "bool":
            return torch.bool
        case "complex32":
            return torch.complex32
        case "complex64":
            return torch.complex64
        case "complex128":
            return torch.complex128
        case "qint8":
            return torch.qint8
        case "quint8":
            return torch.quint8
        case "qint32":
            return torch.qint32
        case "quint4x2":
            return torch.quint4x2
        case "quint2x4":
            return torch.quint2x4
        case "float":
            return torch.float
        case "double":
            return torch.double
        case "half":
            return torch.half
        case "long":
            return torch.long
        case "short":
            return torch.short
        case "cfloat":
            return torch.cfloat
        case "cdouble":
            return torch.cdouble


setattr(torch, "dtypes", dtypes)
