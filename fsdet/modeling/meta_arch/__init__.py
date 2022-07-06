# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .rcnn_sgrn import RCNN_SGRN
from .rcnn_sgrn_copy import RCNN_SGRN_COPY
from .retinanet import RetinaNet
