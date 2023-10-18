# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import wget

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"

"""
    (wget)Download a file from the source URL to the target path if it doesn't already exist.
    
    Args:
        source (str): The URL of the file to download.
        target (str): The local path where the file should be saved.
        
    Returns:
        None
"""
def maybe_download_file(source, target):
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"Downloading {source} to {target}")
        wget.download(source, out=str(target))
        print()

"""
    Construct the full URL for a file given its path relative to the BASE_URL.
    
    Args:
        path (str): The relative path of the file.
        
    Returns:
        str: The full URL of the file.
 """
def get_s3_path(path):
    return f"{BASE_URL}/{path}"


"""
    Construct the local path where a file should be saved.
    
    Args:
        output_dir (str): The base directory where files should be saved.
        path (str): The relative path of the file.
        
    Returns:
        str: The full local path where the file should be saved.
 """
def get_download_path(output_dir, path):
    return os.path.join(output_dir, path)
