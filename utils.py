# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:29:37 2025

@author: abel_
"""

import os

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)