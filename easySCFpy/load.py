import json
import numpy as np
from scipy import sparse
import anndata
import h5py
from typing import Union, Sequence
from anndata._io.specs import read_elem
from anndata._io.h5ad import read_dataframe
from collections.abc import Sequence
from pathlib import Path
from typing import Literal


def read_dense_matrix(h5_group):
    print("Reading dense matrix")
    denses = []
    for key in h5_group.keys():
        sub_matrix = read_elem(h5_group[key])
        denses.append(sub_matrix)
    if h5_group.attrs.get("encoding-type") == "csr_matrix":
        matrix = sparse.vstack(denses, format="csr")
    elif h5_group.attrs.get("encoding-type") == "csc_matrix":
        matrix = sparse.vstack(denses, format="csc")
    else:
        matrix = np.vstack(denses)
    return matrix


def read_matrice_matrix(h5_group, sparse_format):
    matrices = []
    for key in h5_group.keys():
        sub_matrix = read_elem(h5_group[key])
        matrices.append(sub_matrix)

    if sparse_format == sparse.csr_matrix:
        matrices = sparse.vstack(matrices, format="csr")
    else:
        matrices = sparse.vstack(matrices, format="csc")
    return matrices


def h5_to_X(layers, dataName, as_sparse, as_sparse_fmt, chunk_size):
    checkSp = "X" if dataName == "data" else "raw/X"
    if (
        not checkSp in as_sparse
        and layers[dataName].attrs.get("encoding-type") == "array"
    ):
        Data = read_dense_matrix(layers[dataName])
    else:
        Data = read_matrice_matrix(layers[dataName], sparse_format=as_sparse_fmt)
    return Data


def h5_to_misc(h5):
    data_dict = {}

    misc = h5["commands"]

    data_dict = {}

    def parse_group(group, parent_key=""):
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                data = group[key][()]
                data_dict[parent_key + key] = (
                    data if not isinstance(data, str) else data
                )
            elif isinstance(group[key], h5py.Group):
                parse_group(group[key], parent_key + key + "/")

    parse_group(misc)

    reorganized_data_dict = {}
    for key, value in data_dict.items():
        keys = key.split("/")
        current_dict = reorganized_data_dict
        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})
        current_dict[keys[-1]] = value

    def deserialize_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = deserialize_dict(v)
            elif isinstance(v, str):
                try:
                    d[k] = json.loads(v)
                except ValueError:
                    pass
        return d

    reorganized_data_dict = deserialize_dict(reorganized_data_dict)

    return reorganized_data_dict


def h5_to_uns_dict(h5file: h5py.File, path: str) -> dict:
    """
    Recursively reads a dictionary from an h5 file.

    :param h5file: h5py file object
    :param path: path in the h5 file where the data is stored
    :return: dictionary read from the h5 file
    """
    data = {}
    for key in h5file[path].keys():
        new_path = f"{path}/{key}"
        if isinstance(h5file[new_path], h5py.Group):
            data[key] = h5_to_uns_dict(h5file, new_path)
        else:
            item = h5file[new_path][()]
            if isinstance(item, bytes):
                try:
                    item = json.loads(item.decode("utf-8"))
                except json.JSONDecodeError:
                    item = item.decode("utf-8")
            data[key] = item
    return data


def read_h5_to_scanpy(
    filename: Union[str, Path],
    backed: Union[Literal["r"], Literal["r+"], bool, None] = None,
    *,
    as_sparse: Sequence[str] = ("raw/X"),
    as_sparse_fmt: type[sparse.spmatrix] = sparse.csr_matrix,
    chunk_size: int = 6000,
) -> anndata.AnnData:

    if as_sparse_fmt not in (sparse.csr_matrix, sparse.csc_matrix):
        raise NotImplementedError(
            "Dense formats can only be read to CSR or CSC matrices at this time."
        )
    if isinstance(as_sparse, str):
        as_sparse = [as_sparse]
    else:
        as_sparse = list(as_sparse)
    for i in range(len(as_sparse)):
        if as_sparse[i] in {("raw", "X"), "raw.X"}:
            as_sparse[i] = "raw/X"
        elif as_sparse[i] not in {"raw/X", "X"}:
            raise NotImplementedError(
                "Currently only `X` and `raw/X` can be read as sparse."
            )
    with h5py.File(filename, "r") as h5:
        if "var" not in h5.keys():
            raise KeyError("var not found.")
        if "obs" not in h5.keys():
            raise KeyError("obs not found.")
        if "assay" not in h5.keys():
            raise KeyError("assay not found.")
        try:
            layers = h5.get("assay", {}).get("RNA", {}).get("layers", None)
            if layers is not None:
                if "data" in layers.keys():
                    data = h5_to_X(layers, "data", as_sparse, as_sparse_fmt, chunk_size)
                    rawData = h5_to_X(
                        layers, "rawdata", as_sparse, as_sparse_fmt, chunk_size
                    )

                    obs = read_dataframe(h5["obs"])
                    obs = obs if obs.shape[0] != 0 and obs.shape[1] != 0 else None
                    var = read_dataframe(h5["var/var"])
                    var = var if var.shape[0] != 0 and var.shape[1] != 0 else None
                    adata = anndata.AnnData(
                        X=data,
                        obs=obs,
                        var=var,
                    )

                    obs = read_dataframe(h5["obs"])
                    obs = obs if obs.shape[0] != 0 and obs.shape[1] != 0 else None
                    var = read_dataframe(h5["var/rawvar"])
                    var = var if var.shape[0] != 0 and var.shape[1] != 0 else None
                    adata_raw = anndata.AnnData(
                        X=rawData,
                        obs=obs,
                        var=var,
                    )
                    adata.raw = adata_raw
                else:
                    rawData = h5_to_X(
                        layers, "rawdata", as_sparse, as_sparse_fmt, chunk_size
                    )
                    adata = anndata.AnnData(
                        X=rawData,
                        obs=read_dataframe(h5["obs"]),
                        var=read_dataframe(h5["var/rawvar"]),
                    )
            else:
                raise KeyError("Layers not found.")
        except Exception as e:
            raise KeyError("assay not found.")
        if "reductions" in h5.keys():
            adata.obsm = read_elem(h5["reductions"])
        if "graphs" in h5.keys():
            adata.obsp = read_elem(h5["graphs"])
        if "spatial" in h5.keys():
            adata.spatial = read_elem(h5["spatial"])
        if "commands" in h5.keys():
            adata.uns = h5_to_uns_dict(h5, "uns")
    return adata


def loadH5(
    filename: Union[str, Path],
    datatype="scanpy",
    backed: Union[Literal["r"], Literal["r+"], bool, None] = None,
    *,
    as_sparse: Sequence[str] = ("raw/X"),
    as_sparse_fmt: type[sparse.spmatrix] = sparse.csr_matrix,
    chunk_size: int = 6000,
) -> anndata.AnnData:
    if datatype == "scanpy":
        return read_h5_to_scanpy(
            filename,
            backed=backed,
            as_sparse=as_sparse,
            as_sparse_fmt=as_sparse_fmt,
            chunk_size=chunk_size,
        )
    else:
        raise NotImplementedError(f"Datatype {datatype} not supported.")
