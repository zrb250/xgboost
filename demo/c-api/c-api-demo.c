/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file c-api-demo.c
 * \brief A simple example of using xgboost C API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <math.h>

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

int main(int argc, char** argv) {
  int silent = 0;
  int use_gpu = 0;  // set to 1 to use the GPU for training
  
  // load the data
  DMatrixHandle dtrain, dtest;
  //safe_xgboost(XGDMatrixCreateFromFile("../data/agaricus.txt.train", silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile("../data/agaricus.txt.test", silent, &dtest));
  //safe_xgboost(XGDMatrixCreateFromFile("../data/atrain.txt", silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile("../data/atest.txt", silent, &dtest));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/train12.txt", silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/test12.txt", silent, &dtest));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/train21.txt", silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/test21.txt", silent, &dtest));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/train40.txt", silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/test40.txt", silent, &dtest));
  safe_xgboost(XGDMatrixCreateFromFile("../audio-data/train.txt", silent, &dtrain));
  safe_xgboost(XGDMatrixCreateFromFile("../audio-data/test.txt", silent, &dtest));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/trainall.txt", silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile("../audio-data/testall.txt", silent, &dtest));
  
  // create the booster
  BoosterHandle booster;
  DMatrixHandle eval_dmats[2] = {dtrain, dtest};
  safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

  // configure the training
  // available parameters are described here:
  //   https://xgboost.readthedocs.io/en/latest/parameter.html
  //safe_xgboost(XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "hist"));
  if (use_gpu) {
    // set the number of GPUs and the first GPU to use;
    // this is not necessary, but provided here as an illustration
    safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "1"));
    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "0"));
  } else {
    // avoid evaluating objective and metric on a GPU
    safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "0"));
  }

  //safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softmax"));
  //safe_xgboost(XGBoosterSetParam(booster, "objective", "reg:squarederror"));
  safe_xgboost(XGBoosterSetParam(booster, "num_class", "21"));
  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "5"));
  safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "4"));
  safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1"));
  safe_xgboost(XGBoosterSetParam(booster, "eta", "0.05"));
  //safe_xgboost(XGBoosterSetParam(booster, "eval_metric", "mae"));
  safe_xgboost(XGBoosterSetParam(booster, "eval_metric", "mlogloss"));
  
  // train and evaluate for 10 iterations
  int n_trees = 50;
  const char* eval_names[2] = {"train", "test"};
  const char* eval_result = NULL;
  int i = 0;
  for (i = 0; i < n_trees; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
    safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
    printf("%s\n", eval_result);
  }

  // predict
  bst_ulong out_len = 0;
  const float* out_result = NULL;
  
  const float* out_label = NULL;
  int n_print = 10;

  safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result));
  printf("y_pred: ");
  for (i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");

  // print true labels
  safe_xgboost(XGDMatrixGetFloatInfo(dtest, "label", &out_len, &out_label));
  printf("y_test: ");
  for (i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_label[i]);
  }

  int suc = 0;
  int suc1 = 0;
  int suc15 = 0;
  int suc20 = 0;
  float dev = 0;
  for(int j = 0; j < out_len; ++j){
	dev = abs(out_result[j] - out_label[j]);
	if(dev ==0){
		suc = suc + 1;
	}
	if(dev <= 1){
		suc1 = suc1 + 1;
	}
	if(dev <= 1.5){
		suc15 = suc15 + 1;
	}
	if(dev <= 2){
		suc20 = suc20 + 1;
	}
  }

  printf("\n");
  printf("succ:%d, total:%d, acc:%f \n", suc, out_len, (suc*1.0)/out_len);
  printf("succ1:%d, total:%d, acc:%f \n", suc1, out_len, (suc1*1.0)/out_len);
  printf("succ15:%d, total:%d, acc:%f \n", suc15, out_len, (suc15*1.0)/out_len);
  printf("succ20:%d, total:%d, acc:%f \n", suc20, out_len, (suc20*1.0)/out_len);
  printf("\n");

  // free everything`
  safe_xgboost(XGBoosterFree(booster));
  safe_xgboost(XGDMatrixFree(dtrain));
  safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
