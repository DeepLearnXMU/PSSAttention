#! /bin/bash
THEANO_FLAGS="device=gpu4" python test-p-value.py -ds_name 14semeval_laptop -log_name 14semeval_laptop >./log/14semeval_laptop/test_log_info 2>&1 &
