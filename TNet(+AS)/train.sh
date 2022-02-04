#! /bin/bash
THEANO_FLAGS="device=gpu3" python main_total.py -ds_name 14semeval_rest -log_name 14semeval_rest22 >./log/14semeval_rest22/all_log_info 2>&1 &
