/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file c-api-demo.c
 * \brief A simple example of using xgboost C API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

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
  safe_xgboost(XGDMatrixCreateFromFile("../data/audio_train.txt", silent, &dtrain));
  safe_xgboost(XGDMatrixCreateFromFile("../data/audio_test.txt", silent, &dtest));
  
  // create the booster
  BoosterHandle booster;
  DMatrixHandle eval_dmats[2] = {dtrain, dtest};
  safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

  // configure the training
  // available parameters are described here:
  //   https://xgboost.readthedocs.io/en/latest/parameter.html
  safe_xgboost(XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "hist"));
  if (use_gpu) {
    // set the number of GPUs and the first GPU to use;
    // this is not necessary, but provided here as an illustration
    safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "1"));
    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "0"));
  } else {
     //avoid evaluating objective and metric on a GPU
    safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "0"));
  }

  //safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softmax"));
  safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softprob"));
  safe_xgboost(XGBoosterSetParam(booster, "num_class", "17"));
  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "5"));
  safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "2"));
  safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "1"));
  safe_xgboost(XGBoosterSetParam(booster, "eta", "0.5"));
  
  // train and evaluate for 10 iterations
  int n_trees = 3;
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
  int n_print = 10;

  const char* fname="./pxgb.model";
safe_xgboost(XGBoosterSaveModel(booster, fname));
  
  BoosterHandle bsth;
  safe_xgboost(XGBoosterCreate(dtest, 1, &bsth));
  safe_xgboost(XGBoosterLoadModel(bsth,fname))
  safe_xgboost(XGBoosterPredict(bsth, dtest, 0, 0, &out_len, &out_result));
  printf("y_loadmodel: ");
  for (i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");

  safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result));
  printf("y_pred: ");
  for (i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");

  // print true labels
  safe_xgboost(XGDMatrixGetFloatInfo(dtest, "label", &out_len, &out_result));
  printf("y_test: ");
  for (i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");

  // free everything
  safe_xgboost(XGBoosterFree(booster));
  safe_xgboost(XGBoosterFree(bsth));
  safe_xgboost(XGDMatrixFree(dtrain));
  safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
