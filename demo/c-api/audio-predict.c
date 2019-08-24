/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file audio-predict.c
 * \brief A simple example of using xgboost C API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <vector>
#include <sys/time.h>
#include<pthread.h>

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

void formatFeat(std::vector<float> input, float * out)
{
    int len = input.size();
    for (int i = 0; i < len; i++){
        out[i] = input[i];
    }
}


int main(int argc, char** argv) {
  // load the data
  DMatrixHandle  dtest;
  DMatrixHandle  ftest;
  int silent = 0;
  safe_xgboost(XGDMatrixCreateFromFile("../data/audio_train.txt", silent, &dtest));
  
  const int sample_rows = 1;
  std::vector<float> input({499.787,62.4734,26.3046,110.959,191.072,26.3046,45.2967,0.16,1.3125,-1.29511,-2.27628,-3.39993,-1.25392,0.172414,67.4171,0.0,-0.611436,2.8026e-45,0.35522,0.295277,0.131719,0.0845613,360,390.977,0.151429,0.12625,60.0276,88.9758,8,4.50425,2.93,3.22,2,0.7325,4.88428,1.7761,0.35875,1,3,0.0131818,0.0989761,0.0966667,0.0555556,0.062361,3,0.0966667,0.03625,0.0555556,0.062361,3,0.980392,0,2.73038,1.7761});
  float out[input.size()];
  formatFeat(input, out);
  XGDMatrixCreateFromMat((float *) out, sample_rows, input.size(), -1, &ftest);
  // predict
  bst_ulong out_len = 0;
  const float* out_result = NULL;
  int i=0;
  int n_print = 10;

  const char* fname="./pxgb.model";

  BoosterHandle bsth;
  DMatrixHandle eval_dmats[1] = {ftest};
  //safe_xgboost(XGBoosterCreate(eval_dmats, 1, &bsth));
  safe_xgboost(XGBoosterCreate(0, 0, &bsth));
  safe_xgboost(XGBoosterSetParam(bsth, "n_gpus", "0"));
  safe_xgboost(XGBoosterLoadModel(bsth,fname));
  safe_xgboost(XGBoosterPredict(bsth, dtest, 0, 0, &out_len, &out_result));
  printf("y_loadmodel: ");
  for (i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");
  
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

  for(int j=0; j <10; ++j){
  struct timeval sv,ev;
  gettimeofday(&sv,NULL);
  pthread_mutex_lock(&mutex);
  safe_xgboost(XGBoosterPredict(bsth, ftest, 0, 0, &out_len, &out_result));
  pthread_mutex_unlock(&mutex);
  gettimeofday(&ev,NULL);
  printf("timecost:ev:%ld, sv:%ld,  cost:%ld us\n",ev.tv_usec, sv.tv_usec,  ev.tv_usec-sv.tv_usec);
  printf("y_vec: ");
  for (i = 0; i < 1; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");
  }

  // free everything
  safe_xgboost(XGBoosterFree(bsth));
  safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
