# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


class MonoscaledepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MonoScaleDepth options")
        pass

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
